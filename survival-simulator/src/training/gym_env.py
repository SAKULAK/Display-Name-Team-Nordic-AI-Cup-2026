"""
Custom multi-agent training environment wrapping SimulationCore.

Does not implement the plain single-agent gymnasium.Env
step(action) -> obs, reward, terminated, truncated, info API: the simulator
controls a whole (changing) population of agents per tick, sharing one
policy (homogeneous parameter sharing). Instead step()/reset() work on
dicts keyed by agent_id, following the common "__all__" convention for the
population-level episode-done flag (same as RLlib multi-agent envs).
"""
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

# Observation encoding 
# Max number of nearest entities of each type kept in the fixed-size observation
# vector (closest-first, zero-padded if fewer are visible/audible).
K_FRUIT = 6
K_AGENT = 4
K_PREDATOR = 4
K_TREE = 4
K_EDGE = 4

# Rough normalization constants. Taken from environment.py
NORM_DIST = 400.0
NORM_SPEED = 50.0
NORM_AGE = 120.0
NORM_MAX_ENERGY = 1000.0

BIOME_TYPES = ["forest", "grassland", "swamp", "desert", "river"]

# own-state features: energy_frac, age, speed, sprint_speed, hearing_radius,
# vision_angle, vision_range, max_energy, sim_time_frac (episode progress)
OWN_STATE_DIM = 9
# Per-entity-slot feature widths. Angles are encoded as (sin, cos) pairs, not a raw
# linear value, since a linear encoding is discontinuous right where real angles wrap
# around , forcing an awkward, avoidable discontinuity onto exactly
# the case ("something is behind me") that matters most for predator awareness. Each
# slot also ends with an explicit presence flag (1.0 for a real entry, 0.0 for a
# zero-padded empty slot) so "no entity here" isn't otherwise indistinguishable from
# "a real entity sits at angle=0" once angle stops being a single raw number.
FRUIT_SLOT_DIM = 4      # distance, sin(angle), cos(angle), presence
TREE_SLOT_DIM = 4       # distance, sin(angle), cos(angle), presence
EDGE_SLOT_DIM = 4        # distance, sin(angle), cos(angle), presence
AGENT_SLOT_DIM = 6      # distance, sin(angle), cos(angle), sin(rel_dir), cos(rel_dir), presence
PREDATOR_SLOT_DIM = 6   # distance, sin(angle), cos(angle), sin(rel_dir), cos(rel_dir), presence
OBS_DIM = (
    OWN_STATE_DIM
    + len(BIOME_TYPES)
    + K_FRUIT * FRUIT_SLOT_DIM
    + K_AGENT * AGENT_SLOT_DIM
    + K_PREDATOR * PREDATOR_SLOT_DIM
    + K_TREE * TREE_SLOT_DIM
    + K_EDGE * EDGE_SLOT_DIM
)

# Reward shaping
# An agent's own energy change - correctly attributes fruit eating, movement cost, and
# spawn cost to whichever agent actually caused them.
ENERGY_SHAPING_COEF = 0.01

# Potential-based shaping: reward closing distance to the nearest currently-known
# fruit. Without a good "it's worth moving" signal, movement is pure cost 
# #(walking/sprinting/turning all drain energy) with only a rare, delayed
# payoff from actually reaching food. Calibrated against Environment's own movement
# cost formula,: sprinting costs energy on a much steeper curve than walking #
# (jumps from 0.05/unit to 0.5/unit past speed), so a coefficient too
# small makes "sprint toward food" net-negative even though "walk toward food" is
# net-positive - that is hard to learn. 1.5 keeps net reward for
# closing distance flat and comfortably positive across the full
# walk-to-sprint range, with margin to spare even ignoring SEARCH_COEF's contribution.
FRUIT_APPROACH_COEF = 1.5

# Reward for how far an agent's actual position has drifted from a slow-following
# exponential moving average (EMA) of its own recent position (tracked internally via
# the raw x/y on the underlying Agent object - reward-only information,
# never exposed in the agent's own observation vector). Encourages genuine searching:
# pure oscillation or staying in one place means position stays close to wherever the
# agent's been dwelling, so the EMA catches up and the gap - and the reward - stays
# near zero. This is so these fuck ass agents get off their asses and move around instead
#of just looking
SEARCH_COEF = 0.1

# How fast the EMA reference catches up to the agent's actual position each tick
# (exponential decay rate, not a hard window size). Smaller = slower to catch up =
# rewards longer, more sustained excursions before the signal fades.
SEARCH_EMA_ALPHA = 0.02

# Mirror of FRUIT_APPROACH_COEF, sign flipped: reward increasing distance to the
# nearest known predator, penalize closing it. More directly attributable
# than relying solely on DEATH_PENALTY
PREDATOR_AVOID_COEF = 1.5

# Flat penalty on top of that tick's other rewards for the specific agent that died.
DEATH_PENALTY = -2.0

# Bonus for the parent on the tick a spawn actually succeeds (spawn_agent=True AND
# pre-step energy > 100 Net effect of a "safe" successful spawn (see SPAWN_SAFETY_MARGIN below)
#  is  +2.0 - 1.0 = +1.0. # SPAWN_REWARD is scaled by how much energy the parent has left after 
# paying the 100 cost,
SPAWN_REWARD = 2.0

# SPAWN_REWARD is scaled by how much energy the parent has left after paying the 100
# cost, ramping linearly from 0 (0 energy left) to full credit (>= this much energy
# left), rather than a flat bonus regardless of outcome - so a spawn that leaves the
# parent one bad tick from starving isn't rewarded as generously as a safe one.
SPAWN_SAFETY_MARGIN = 50.0

# Reward for how aligned this tick's actual displacement is with the previous tick's,
# scaled by how far it moved this tick. Sustained, committed movement in roughly one
# direction scores high; a random walk's consecutive directions are independent by
# construction, so cos_similarity averages to 0 over many ticks - structurally safer
# than SEARCH_COEF's cumulative EMA-gap approach, which is what let a too-large
# SEARCH_COEF get farmed by pure diffusion (see its own comment). This doesn't reward
# accumulated drift at all, only genuine tick-to-tick directional consistency, so
# there's no equivalent drift to over-reward in the first place. Complements
# SEARCH_COEF (a slower, memory-based "have I actually gotten anywhere over time"
# signal) with an immediate "am I currently committing to a direction, not
# dithering" one - also directly discourages the jittery oscillation/hesitation
# pattern reported during training, since reversing direction tick-to-tick scores
# negative (cos_similarity < 0), not just neutral.
MOMENTUM_COEF = 0.5

# Distance within which Predator.step() charges the closest agent directly
# regardless of whether that agent is facing it (mirrors hearing_radius=60 * 1.5 from
# predator.py exactly - see PREDATOR_FACE_COEF below). Facing only changes predator
# behavior beyond this range, so it's the threshold for PREDATOR_FACE_COEF's own gate.
PREDATOR_CHARGE_RANGE = 90.0

# Potential-based: reward reducing the angle to a still-distant known predator (i.e.
# turning to face it), penalize increasing it. Zero once the predator is within
# PREDATOR_CHARGE_RANGE, where it no longer matters (see below) - this is deliberately
# NOT a general "always face predators" reward, since that would fight
# PREDATOR_AVOID_COEF's flee incentive once a predator is actually closing in.
#
# This exploits a real branch in Predator.step()'s own scripted AI (predator.py):
# it only charges the closest agent directly when that agent isn't facing it (or
# it's already within PREDATOR_CHARGE_RANGE); otherwise it does a slower, more
# cautious flanking maneuver instead. The angle that decision is based on
# (`agent_looking_dir`, computed by the predator from the agent's own .direction) is
# mathematically identical to the angle the agent itself already observes toward that
# predator (its own `angle` field in the Predator observation slot) - both reduce to
# arctan2(predator.y - agent.y, predator.x - agent.x) - agent.direction, just computed
# from opposite sides of the same relationship. So the agent already has direct access
# to the exact quantity that decides the predator's behavior; this reward just makes
# using it worth learning.
#
# Telescoping potential shaping like FRUIT_APPROACH_COEF, so it's immune to
# path-based farming: oscillating the facing angle back and forth nets to zero over
# any full cycle (sum of per-tick deltas = final |angle| - initial |angle|,
# regardless of path), so only genuinely bringing a predator into view and holding it
# there accumulates reward - the same guarantee that makes fruit_approach_reward safe
# against distance oscillation.
PREDATOR_FACE_COEF = 0.5


def _bucket(entries, k: int, sort_key, feature_fn, n_features: int) -> np.ndarray:
    entries_sorted = sorted(entries, key=sort_key)[:k]
    vec = np.zeros(k * n_features, dtype=np.float32)
    for i, entry in enumerate(entries_sorted):
        vec[i * n_features:(i + 1) * n_features] = feature_fn(entry)
    return vec


def _dist_angle(obs: dict) -> np.ndarray:
    angle = obs["angle"]
    return np.array(
        [obs["distance"] / NORM_DIST, np.sin(angle), np.cos(angle), 1.0],
        dtype=np.float32,
    )


def _dist_angle_dir(obs: dict) -> np.ndarray:
    angle = obs["angle"]
    rel_dir = obs["rel_dir"]
    return np.array(
        [obs["distance"] / NORM_DIST, np.sin(angle), np.cos(angle),
         np.sin(rel_dir), np.cos(rel_dir), 1.0],
        dtype=np.float32,
    )


def _edge_midpoint_dist(obs: dict) -> float:
    (sx, sy), (ex, ey) = obs["coords"]
    return float(np.hypot((sx + ex) / 2.0, (sy + ey) / 2.0))


def _edge_feat(obs: dict) -> np.ndarray:
    (sx, sy), (ex, ey) = obs["coords"]
    mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
    dist = np.hypot(mx, my)
    angle = np.arctan2(my, mx)
    return np.array([dist / NORM_DIST, np.sin(angle), np.cos(angle), 1.0], dtype=np.float32)


def _nearest_fruit_distance(status: dict):
    """Raw (unnormalized) distance to the closest currently seen/heard fruit, or None."""
    fruit_distances = [o["distance"] for o in status["observations"] if o["type"] == "Fruit"]
    return min(fruit_distances) if fruit_distances else None


def _nearest_predator_distance(status: dict):
    """Raw (unnormalized) distance to the closest currently seen/heard predator, or None."""
    predator_distances = [o["distance"] for o in status["observations"] if o["type"] == "Predator"]
    return min(predator_distances) if predator_distances else None


def _nearest_predator_angle(status: dict):
    """Angle (relative to own facing) to the closest currently seen/heard predator, by
    the same nearest-by-distance predator _nearest_predator_distance tracks, or None."""
    predators = [o for o in status["observations"] if o["type"] == "Predator"]
    if not predators:
        return None
    return min(predators, key=lambda o: o["distance"])["angle"]


# Reward formulas (pure, unit-testable) 

def fruit_approach_reward(prev_dist, curr_dist) -> float:
    """Potential-based: positive for closing the gap to the nearest known fruit,
    negative for increasing it. 0.0 if not tracked on both sides of the tick."""
    if prev_dist is None or curr_dist is None:
        return 0.0
    return FRUIT_APPROACH_COEF * (prev_dist - curr_dist) / NORM_DIST


def predator_avoid_reward(prev_dist, curr_dist) -> float:
    """Mirror of fruit_approach_reward, sign flipped."""
    if prev_dist is None or curr_dist is None:
        return 0.0
    return PREDATOR_AVOID_COEF * (curr_dist - prev_dist) / NORM_DIST


def search_reward_and_ref(curr_pos: Tuple[float, float], ref_pos: Tuple[float, float]):
    """Reward for how far curr_pos has drifted from the slow-following EMA reference
    ref_pos, plus the updated reference for next tick. See SEARCH_COEF/SEARCH_EMA_ALPHA."""
    gap = float(np.hypot(curr_pos[0] - ref_pos[0], curr_pos[1] - ref_pos[1]))
    reward = SEARCH_COEF * min(gap, NORM_DIST) / NORM_DIST
    new_ref = (
        ref_pos[0] + SEARCH_EMA_ALPHA * (curr_pos[0] - ref_pos[0]),
        ref_pos[1] + SEARCH_EMA_ALPHA * (curr_pos[1] - ref_pos[1]),
    )
    return reward, new_ref


def momentum_reward(prev_disp, curr_disp) -> float:
    """See MOMENTUM_COEF. Cosine similarity between this tick's and last tick's
    displacement vectors, scaled by how far this tick actually moved. 0.0 if either
    tick had ~no movement (alignment is undefined for a near-zero vector, and
    shouldn't be rewarded or punished either way)."""
    if prev_disp is None or curr_disp is None:
        return 0.0
    prev_mag = float(np.hypot(prev_disp[0], prev_disp[1]))
    curr_mag = float(np.hypot(curr_disp[0], curr_disp[1]))
    if prev_mag < 1e-6 or curr_mag < 1e-6:
        return 0.0
    cos_similarity = (prev_disp[0] * curr_disp[0] + prev_disp[1] * curr_disp[1]) / (prev_mag * curr_mag)
    return MOMENTUM_COEF * cos_similarity * (curr_mag / NORM_DIST)


def predator_face_reward(prev_angle, curr_angle, curr_distance) -> float:
    """See PREDATOR_FACE_COEF. Potential-based on |angle| to the nearest known
    predator, gated off once that predator is within PREDATOR_CHARGE_RANGE (where
    facing no longer affects its behavior) or not tracked on both sides of the tick."""
    if prev_angle is None or curr_angle is None or curr_distance is None:
        return 0.0
    if curr_distance <= PREDATOR_CHARGE_RANGE:
        return 0.0
    return PREDATOR_FACE_COEF * (abs(prev_angle) - abs(curr_angle)) / np.pi


def spawn_reward(pre_spawn_energy: float) -> float:
    """SPAWN_REWARD scaled by how much energy remains after the 100 spawn cost, or
    0.0 if spawning wasn't actually possible (mirrors Environment.agent_step's own
    `energy > 100` gate exactly)."""
    if pre_spawn_energy <= 100:
        return 0.0
    remaining_energy = pre_spawn_energy - 100
    safety_frac = float(np.clip(remaining_energy / SPAWN_SAFETY_MARGIN, 0.0, 1.0))
    return SPAWN_REWARD * safety_frac


def encode_observation(status: dict, sim_time_frac: float) -> np.ndarray:
    """Turn one agent's ObservationResponse dict into a fixed-size float32 vector.

    sim_time_frac: current episode time / max_time, clipped to [0, 1] - gives the
    agent a sense of how far into the episode it is, since nothing in status itself
    (ObservationResponse has no time field) exposes that.
    """
    own = np.array([
        status["energy"] / max(status["max_energy"], 1e-6),
        min(status["age"] / NORM_AGE, 2.0),
        status["speed"] / NORM_SPEED,
        status["sprint_speed"] / NORM_SPEED,
        status["hearing_radius"] / NORM_DIST,
        status["vision_angle"] / np.pi,
        status["vision_range"] / NORM_DIST,
        status["max_energy"] / NORM_MAX_ENERGY,
        sim_time_frac,
    ], dtype=np.float32)

    biome_onehot = np.zeros(len(BIOME_TYPES), dtype=np.float32)
    if status["biome"] in BIOME_TYPES:
        biome_onehot[BIOME_TYPES.index(status["biome"])] = 1.0

    fruits, agents, predators, trees, edges = [], [], [], [], []
    for obs in status["observations"]:
        obs_type = obs["type"]
        if obs_type == "Fruit":
            fruits.append(obs)
        elif obs_type == "Agent":
            agents.append(obs)
        elif obs_type == "Predator":
            predators.append(obs)
        elif obs_type == "Tree":
            trees.append(obs)
        elif obs_type == "Edge":
            edges.append(obs)

    fruit_vec = _bucket(fruits, K_FRUIT, lambda o: o["distance"], _dist_angle, FRUIT_SLOT_DIM)
    agent_vec = _bucket(agents, K_AGENT, lambda o: o["distance"], _dist_angle_dir, AGENT_SLOT_DIM)
    predator_vec = _bucket(predators, K_PREDATOR, lambda o: o["distance"], _dist_angle_dir, PREDATOR_SLOT_DIM)
    tree_vec = _bucket(trees, K_TREE, lambda o: o["distance"], _dist_angle, TREE_SLOT_DIM)
    edge_vec = _bucket(edges, K_EDGE, _edge_midpoint_dist, _edge_feat, EDGE_SLOT_DIM)

    return np.concatenate([own, biome_onehot, fruit_vec, agent_vec, predator_vec, tree_vec, edge_vec])


def decode_action(
    continuous_action: np.ndarray, spawn_action: float, sprint_speed: float
) -> Tuple[float, float, float, bool]:
    """Map raw network output (as sampled by ActorCritic) to simulator action fields."""
    distance_frac = float(np.clip(continuous_action[0], 0.0, 1.0))
    move_distance = distance_frac * sprint_speed
    move_direction = float(np.arctan2(continuous_action[1], continuous_action[2]))  # (sin, cos) -> angle
    turn_angle = float(np.clip(continuous_action[3], -np.pi, np.pi))
    spawn_agent = bool(spawn_action >= 0.5)
    return move_distance, move_direction, turn_angle, spawn_agent


class SurvivalEnv(gym.Env):
    """Multi-agent training wrapper around SimulationCore. See module docstring."""

    def __init__(
        self,
        env_width: int = 1600,
        env_height: int = 1200,
        chunk_size: int = 400,
        starting_agents: int = 5,
        #was 0 at the start
        starting_predators: int = 1,
        starting_fruits=None,
        starting_trees: int = 50,
        dt: float = 1 / 10,
        max_time: float = 3000.0,
        seed: int = None,
    ):
        super().__init__()
        # Resolve the fruit/tree starting counts to concrete numbers now (rather than
        # leaving starting_fruits=None to be resolved later by SimulationCore) so
        # set_difficulty() below has a fixed baseline to scale from.
        base_starting_fruits = starting_fruits if starting_fruits is not None else env_width // 50
        self._base_starting_fruits = base_starting_fruits
        self._base_starting_trees = starting_trees

        self._sim_kwargs = dict(
            env_width=env_width,
            env_height=env_height,
            chunk_size=chunk_size,
            starting_agents=starting_agents,
            starting_predators=starting_predators,
            starting_fruits=base_starting_fruits,
            starting_trees=starting_trees,
            dt=dt,
        )
        self._seed = seed
        self.max_time = max_time

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "continuous": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "spawn": gym.spaces.Discrete(2),
        })

        self.sim: SimulationCore = None
        self._last_status: Dict[int, dict] = {}
        self._prev_energy: Dict[int, float] = {}
        self._prev_fruit_dist: Dict[int, float] = {}
        self._prev_predator_dist: Dict[int, float] = {}
        self._search_ref_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_position: Dict[int, Tuple[float, float]] = {}
        self._prev_displacement: Dict[int, Tuple[float, float]] = {}
        self._prev_predator_angle: Dict[int, float] = {}
        self._predator_speed_mult = 1.0

    def set_difficulty(self, fruit_mult: float = 1.0, tree_mult: float = 1.0, predator_speed_mult: float = 1.0):
        """
        Scale starting fruit/tree counts relative to the env's base config, and
        predator speed/sprint_speed by predator_speed_mult (applied on top of each
        predator's own already-mutated speed, at reset() time) - used for a simple
        curriculum (easier early, annealed up to full difficulty). Predator count
        itself is left alone (starting_predators): a present-but-slow predator still
        lets agents learn to notice and react to it from the start, which an
        absent-then-suddenly-real predator wouldn't."""
        self._sim_kwargs["starting_fruits"] = max(1, int(self._base_starting_fruits * fruit_mult))
        self._sim_kwargs["starting_trees"] = max(1, int(self._base_starting_trees * tree_mult))
        self._predator_speed_mult = float(np.clip(predator_speed_mult, 0.0, 1.0))

    def reset(self, *, seed: int = None, options: dict = None):
        actual_seed = seed if seed is not None else self._seed
        self.sim = SimulationCore(seed=actual_seed, **self._sim_kwargs)

        if self._predator_speed_mult != 1.0:
            for predator in self.sim.env.predators:
                predator.speed *= self._predator_speed_mult
                predator.sprint_speed *= self._predator_speed_mult

        self._last_status = {
            agent.agent_id: self.sim.env.get_agent_state(agent.agent_id)
            for agent in self.sim.env.agents
        }
        self._prev_energy = {aid: status["energy"] for aid, status in self._last_status.items()}
        self._prev_fruit_dist = {aid: _nearest_fruit_distance(status) for aid, status in self._last_status.items()}
        self._prev_predator_dist = {
            aid: _nearest_predator_distance(status) for aid, status in self._last_status.items()
        }
        self._prev_predator_angle = {
            aid: _nearest_predator_angle(status) for aid, status in self._last_status.items()
        }
        self._search_ref_pos = {agent.agent_id: (agent.x, agent.y) for agent in self.sim.env.agents}
        self._prev_position = {agent.agent_id: (agent.x, agent.y) for agent in self.sim.env.agents}
        self._prev_displacement = {agent.agent_id: (0.0, 0.0) for agent in self.sim.env.agents}

        sim_time_frac = min(self.sim.env.time / self.max_time, 1.0)
        obs = {aid: encode_observation(status, sim_time_frac) for aid, status in self._last_status.items()}
        return obs, {}

    def step(self, actions: Dict[int, Tuple[np.ndarray, float]]):
        action_requests = []
        spawn_reward_for: Dict[int, float] = {}
        for agent_id, (continuous_action, spawn_action) in actions.items():
            status = self._last_status.get(agent_id)
            if status is None:  # agent died/was unknown before this tick was decided
                continue
            move_distance, move_direction, turn_angle, spawn_agent = decode_action(
                continuous_action, spawn_action, status["sprint_speed"]
            )
            if spawn_agent:
                spawn_reward_for[agent_id] = spawn_reward(status["energy"])
            action_requests.append((agent_id, ActionRequest(
                agent_id=agent_id,
                move_distance=move_distance,
                move_direction=move_direction,
                turn_angle=turn_angle,
                spawn_agent=spawn_agent,
            )))

        prev_alive_ids = set(self._last_status.keys())
        prev_energy = self._prev_energy
        prev_fruit_dist = self._prev_fruit_dist
        prev_predator_dist = self._prev_predator_dist
        prev_predator_angle = self._prev_predator_angle
        prev_search_ref = self._search_ref_pos
        prev_position = self._prev_position
        prev_displacement = self._prev_displacement

        state = self.sim.step(action_requests)

        alive_status = {s["agent_id"]: s for s in state["observations"]}
        alive_ids = set(alive_status.keys())
        died_ids = prev_alive_ids - alive_ids
        time_up = self.sim.env.time >= self.max_time
        sim_time_frac = min(self.sim.env.time / self.max_time, 1.0)

        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        updated_search_ref: Dict[int, Tuple[float, float]] = {}
        updated_position: Dict[int, Tuple[float, float]] = {}
        updated_displacement: Dict[int, Tuple[float, float]] = {}

        for agent_id in alive_ids:
            status = alive_status[agent_id]
            energy_delta = status["energy"] - prev_energy.get(agent_id, status["energy"])

            prev_dist = prev_fruit_dist.get(agent_id)
            curr_dist = _nearest_fruit_distance(status)
            fruit_reward = fruit_approach_reward(prev_dist, curr_dist)

            prev_pred_dist = prev_predator_dist.get(agent_id)
            curr_pred_dist = _nearest_predator_distance(status)
            pred_reward = predator_avoid_reward(prev_pred_dist, curr_pred_dist)

            prev_pred_angle = prev_predator_angle.get(agent_id)
            curr_pred_angle = _nearest_predator_angle(status)
            face_reward = predator_face_reward(prev_pred_angle, curr_pred_angle, curr_pred_dist)

            agent_obj = self.sim.env.agents_dict.get(agent_id)
            curr_pos = (agent_obj.x, agent_obj.y) if agent_obj is not None else None
            ref_pos = prev_search_ref.get(agent_id, curr_pos)
            if curr_pos is not None and ref_pos is not None:
                explore_reward, updated_search_ref[agent_id] = search_reward_and_ref(curr_pos, ref_pos)
            else:
                explore_reward = 0.0

            if curr_pos is not None:
                prev_pos = prev_position.get(agent_id, curr_pos)
                curr_disp = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                move_reward = momentum_reward(prev_displacement.get(agent_id), curr_disp)
                updated_position[agent_id] = curr_pos
                updated_displacement[agent_id] = curr_disp
            else:
                move_reward = 0.0

            agent_spawn_reward = spawn_reward_for.get(agent_id, 0.0)

            rewards[agent_id] = (
                ENERGY_SHAPING_COEF * energy_delta
                + fruit_reward
                + explore_reward
                + pred_reward
                + face_reward
                + move_reward
                + agent_spawn_reward
            )
            obs[agent_id] = encode_observation(status, sim_time_frac)
            terminated[agent_id] = False
            truncated[agent_id] = time_up
            infos[agent_id] = {}

        for agent_id in died_ids:
            # No obs entry: the agent is gone and must not be carried forward into
            # next tick's action selection Still credit a successful spawn even if the same
            # tick's predator/energy check killed this agent right after.
            agent_spawn_reward = spawn_reward_for.get(agent_id, 0.0)
            rewards[agent_id] = DEATH_PENALTY + agent_spawn_reward
            terminated[agent_id] = True
            truncated[agent_id] = False
            infos[agent_id] = {}

        self._last_status = alive_status
        self._prev_energy = {aid: s["energy"] for aid, s in alive_status.items()}
        self._prev_fruit_dist = {aid: _nearest_fruit_distance(s) for aid, s in alive_status.items()}
        self._prev_predator_dist = {aid: _nearest_predator_distance(s) for aid, s in alive_status.items()}
        self._prev_predator_angle = {aid: _nearest_predator_angle(s) for aid, s in alive_status.items()}
        self._search_ref_pos = updated_search_ref
        self._prev_position = updated_position
        self._prev_displacement = updated_displacement

        terminated["__all__"] = state["num_agents"] == 0
        truncated["__all__"] = time_up
        infos["__all__"] = {
            "score": state["score"],
            "sim_time": state["sim_time"],
            "num_agents": state["num_agents"],
        }

        return obs, rewards, terminated, truncated, infos
