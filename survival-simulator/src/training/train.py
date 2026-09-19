"""
PPO training loop for the survival simulator's shared homogeneous agent policy.

Each currently-alive agent is treated as an independent rollout worker sharing
one ActorCritic. Because the population changes size every tick (agents spawn
and die), transitions are collected per-agent-id into separate Trajectory
buffers, and a fresh set of buffers is started whenever the whole-population
episode ends ("__all__" terminated/truncated) mid-rollout - this keeps GAE
from being computed across an episode boundary or across two different
agents that happen to reuse the same freshly-reset agent_id.
"""
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.gym_env import OBS_DIM
from src.training.model import ActorCritic
from src.training.vec_env import SubprocVecSurvivalEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ENVS = 8                 # parallel SurvivalEnv instances collected into each rollout
TOTAL_UPDATES = 10000      # number of PPO update iterations (training is fully resumable via checkpoints, so this is just a cap)
ROLLOUT_TICKS = 256        # env ticks collected per rollout, per env (x N_ENVS transitions/update)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VALUE_COEF = 0.5
# Entropy bonus coefficient, linearly annealed from START to END across training so
# exploration is encouraged early (when the policy needs to discover that moving
# toward food pays off) and settles down later once behavior is established.
ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.005
ENTROPY_ANNEAL_UPDATES = 500
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
MAX_GRAD_NORM = 0.5

# Curriculum: start with extra fruit/tree density so food is easy to find while the
# fruit-approach reward shaping is still bootstrapping "moving toward food is good"
# behavior, then anneal down to the environment's normal difficulty.
# SPAWN_REWARD lets the colony grow well beyond what the *declining* food supply can
# support
CURRICULUM_UPDATES = 400
CURRICULUM_FRUIT_MULT_START = 1.5
CURRICULUM_TREE_MULT_START = 1.3

CHECKPOINT_PATH = "checkpoints/policy.pt"
CHECKPOINT_EVERY = 20


@dataclass
class Trajectory:
    obs: List[np.ndarray] = field(default_factory=list)
    continuous_action: List[np.ndarray] = field(default_factory=list)
    spawn_action: List[float] = field(default_factory=list)
    log_prob: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)


def collect_rollout(
    vec_env: SubprocVecSurvivalEnv,
    model: ActorCritic,
    obs_list: List[Dict[int, np.ndarray]],
    num_ticks: int,
):
    """
    Step vec_env's envs for num_ticks ticks, batching every env's living agents
    into one forward pass per tick, then dispatching all envs' step() calls to
    their worker processes together (see SubprocVecSurvivalEnv) so the actual
    per-env simulation work happens concurrently across CPU cores instead of
    one env at a time.

    Returns a list of per-env "segments" - each a (trajectories, terminated,
    final_obs) tuple covering one uninterrupted stretch of one env's simulation (a
    new segment starts whenever that env's whole population dies out or the time
    limit is hit and it resets - this keeps GAE from being computed across an
    episode boundary or across two different agents that reuse the same freshly
    -reset agent_id within the same env).
    """
    n_envs = vec_env.n_envs
    trajectories: List[Dict[int, Trajectory]] = [dict() for _ in range(n_envs)]
    finished_terminated: List[Dict[int, bool]] = [dict() for _ in range(n_envs)]
    segments = []
    episode_summaries: List[dict] = []  # one "__all__" info snapshot per episode that ended this rollout
    latest_info_per_env: List[dict] = [{} for _ in range(n_envs)]  # each env's most recent "__all__" info

    for _ in range(num_ticks):
        per_env_agent_ids = [list(obs.keys()) for obs in obs_list]
        obs_rows = [obs_list[e][aid] for e in range(n_envs) for aid in per_env_agent_ids[e]]

        if not obs_rows:  # every env's population happens to be empty this tick (shouldn't normally occur)
            obs_list = vec_env.reset()
            continue

        obs_batch = torch.as_tensor(np.stack(obs_rows), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            continuous_action, spawn_action, log_prob, _, value = model.get_action_and_value(obs_batch)

        continuous_np = continuous_action.cpu().numpy()
        spawn_np = spawn_action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.cpu().numpy()

        # Slice the flat batch back out per env, and build every env's actions dict
        # up front so vec_env.step() can dispatch to all worker processes at once.
        local_slices: List[slice] = []
        actions_per_env: List[Dict[int, Tuple[np.ndarray, float]]] = []
        offset = 0
        for env_idx in range(n_envs):
            agent_ids = per_env_agent_ids[env_idx]
            n = len(agent_ids)
            local = slice(offset, offset + n)
            local_slices.append(local)
            offset += n
            local_continuous, local_spawn = continuous_np[local], spawn_np[local]
            actions_per_env.append({aid: (local_continuous[i], local_spawn[i]) for i, aid in enumerate(agent_ids)})

        step_results = vec_env.step(actions_per_env)

        for env_idx in range(n_envs):
            agent_ids = per_env_agent_ids[env_idx]
            n = len(agent_ids)
            if n == 0:  # defensive: shouldn't normally happen, see reset-on-"__all__" below
                obs_list[env_idx], _ = vec_env.reset_one(env_idx)
                continue

            local = local_slices[env_idx]
            local_continuous, local_spawn = continuous_np[local], spawn_np[local]
            local_log_prob, local_value = log_prob_np[local], value_np[local]

            next_obs, rewards, terminated, truncated, infos = step_results[env_idx]
            latest_info_per_env[env_idx] = infos["__all__"]

            env_traj, env_term = trajectories[env_idx], finished_terminated[env_idx]
            for i, aid in enumerate(agent_ids):
                traj = env_traj.setdefault(aid, Trajectory())
                traj.obs.append(obs_list[env_idx][aid])
                traj.continuous_action.append(local_continuous[i])
                traj.spawn_action.append(local_spawn[i])
                traj.log_prob.append(local_log_prob[i])
                traj.value.append(local_value[i])
                traj.reward.append(rewards[aid])
                if terminated[aid]:
                    env_term[aid] = True

            obs_list[env_idx] = next_obs

            if terminated.get("__all__") or truncated.get("__all__"):
                episode_summaries.append(infos["__all__"])
                segments.append((env_traj, env_term, next_obs))
                trajectories[env_idx], finished_terminated[env_idx] = {}, {}
                obs_list[env_idx], _ = vec_env.reset_one(env_idx)

    for env_idx in range(n_envs):
        segments.append((trajectories[env_idx], finished_terminated[env_idx], obs_list[env_idx]))

    return segments, episode_summaries, latest_info_per_env, obs_list


def compute_gae(trajectory: Trajectory, bootstrap_value: float):
    """Walk a single agent's trajectory backward to compute GAE advantages/returns."""
    rewards = trajectory.reward
    values = trajectory.value + [bootstrap_value]

    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + np.array(values[:-1], dtype=np.float32)
    return advantages, returns


def bootstrap_value_for(model: ActorCritic, last_obs) -> float:
    if last_obs is None:
        return 0.0
    with torch.no_grad():
        obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        _, _, _, _, value = model.get_action_and_value(obs_t)
    return value.item()


def build_batch(model: ActorCritic, segments):
    obs_buf, cont_act_buf, spawn_act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []

    for trajectories, terminated, final_obs in segments:
        for agent_id, traj in trajectories.items():
            if len(traj.reward) == 0:
                continue

            if terminated.get(agent_id):
                bootstrap_value = 0.0  # agent died: no future reward to bootstrap
            else:
                bootstrap_value = bootstrap_value_for(model, final_obs.get(agent_id))

            advantages, returns = compute_gae(traj, bootstrap_value)

            obs_buf.extend(traj.obs)
            cont_act_buf.extend(traj.continuous_action)
            spawn_act_buf.extend(traj.spawn_action)
            logp_buf.extend(traj.log_prob)
            adv_buf.extend(advantages.tolist())
            ret_buf.extend(returns.tolist())

    return (
        torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.array(cont_act_buf), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.array(spawn_act_buf), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.array(adv_buf), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.array(ret_buf), dtype=torch.float32, device=DEVICE),
    )


def ppo_update(model: ActorCritic, optimizer: optim.Optimizer, batch, entropy_coef: float):
    obs, cont_actions, spawn_actions, old_log_probs, advantages, returns = batch
    n = obs.shape[0]
    if n == 0:
        return

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = np.arange(n)

    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb_idx = torch.as_tensor(indices[start:start + MINIBATCH_SIZE], device=DEVICE)

            _, _, new_log_probs, entropy, values = model.get_action_and_value(
                obs[mb_idx], action=(cont_actions[mb_idx], spawn_actions[mb_idx])
            )

            ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
            mb_adv = advantages[mb_idx]

            policy_loss_unclipped = -mb_adv * ratio
            policy_loss_clipped = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

            value_loss = 0.5 * (values - returns[mb_idx]).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def save_checkpoint(model: ActorCritic, optimizer: optim.Optimizer, update: int):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
    }, CHECKPOINT_PATH)


def load_checkpoint(model: ActorCritic, optimizer: optim.Optimizer) -> int:
    """Load model/optimizer state if a checkpoint exists. Returns the update to resume from."""
    if not os.path.exists(CHECKPOINT_PATH):
        return 1

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_update = checkpoint["update"] + 1
    print(f"resumed from {CHECKPOINT_PATH} at update {start_update}")
    return start_update


def train():
    random.seed(0)
    torch.manual_seed(0)

    # Created before the model/optimizer so no CUDA context exists yet when worker
    # processes are spawned (relevant on an HPC box with a GPU)
    vec_env = SubprocVecSurvivalEnv(N_ENVS)
    model = ActorCritic(obs_dim=OBS_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    start_update = load_checkpoint(model, optimizer)

    obs_list = vec_env.reset()

    total_episodes = 0

    try:
        for update in range(start_update, TOTAL_UPDATES + 1):
            start_time = time.time()

            # Anneal curriculum difficulty and exploration entropy. Only affects envs
            # that reset from here on - an in-progress episode keeps its own settings.
            curriculum_progress = min(1.0, update / CURRICULUM_UPDATES)
            fruit_mult = CURRICULUM_FRUIT_MULT_START + (1.0 - CURRICULUM_FRUIT_MULT_START) * curriculum_progress
            tree_mult = CURRICULUM_TREE_MULT_START + (1.0 - CURRICULUM_TREE_MULT_START) * curriculum_progress
            vec_env.set_difficulty(fruit_mult, tree_mult)

            entropy_progress = min(1.0, update / ENTROPY_ANNEAL_UPDATES)
            entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * entropy_progress

            segments, episode_summaries, latest_info_per_env, obs_list = collect_rollout(
                vec_env, model, obs_list, ROLLOUT_TICKS
            )

            batch = build_batch(model, segments)
            ppo_update(model, optimizer, batch, entropy_coef)

            n_transitions = batch[0].shape[0]
            all_rewards = [
                r for trajectories, _, _ in segments for traj in trajectories.values() for r in traj.reward
            ]
            mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

            for summary in episode_summaries:
                total_episodes += 1
                print(
                    f"  [episode {total_episodes} finished] score={summary['score']:.2f} "
                    f"agents_alive={summary['num_agents']} sim_time={summary['sim_time']:.1f}s"
                )

            # Aggregate across the N_ENVS parallel envs for one representative progress line.
            known = [info for info in latest_info_per_env if info]
            mean_score = float(np.mean([info["score"] for info in known])) if known else 0.0
            mean_agents = float(np.mean([info["num_agents"] for info in known])) if known else 0.0
            mean_sim_time = float(np.mean([info["sim_time"] for info in known])) if known else 0.0

            print(
                f"update {update:5d} | avg score {mean_score:7.2f} | avg agents {mean_agents:4.1f} | "
                f"avg sim_time {mean_sim_time:7.1f}s | transitions {n_transitions:5d} | "
                f"mean_reward {mean_reward:+.4f} | entropy_coef {entropy_coef:.4f} | "
                f"fruit_mult {fruit_mult:.2f} | {time.time() - start_time:.1f}s"
            )

            if update % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, update)
                print(f"saved checkpoint to {CHECKPOINT_PATH}")

        save_checkpoint(model, optimizer, TOTAL_UPDATES)
    finally:
        vec_env.close()


if __name__ == "__main__":
    train()
