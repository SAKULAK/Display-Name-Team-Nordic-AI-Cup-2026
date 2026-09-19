"""User-run deterministic or stochastic evaluation. Importing starts nothing."""
import argparse
import csv
from pathlib import Path
import statistics
import time


def evaluate_episode(policy, config, seed, horizon, render=False, deadline=float("inf"),
                     policy_mode="deterministic", policy_seed=0):
    from rl.env_wrapper import SurvivalEnv
    from rl.execution import PolicyExecution
    execution = PolicyExecution(policy, policy_mode, policy_seed)
    env = SurvivalEnv(config)
    packet = env.reset(seed, horizon)
    viewer = None
    if render:
        from rl.render import Viewer
        viewer = Viewer(env.core.env, seed)
    training = policy.training
    policy.eval()
    complete = False
    try:
        while time.monotonic() < deadline:
            if viewer and not viewer.draw(env.core.env):
                break
            actions, _ = execution.act(packet["observations"])
            result = env.begin(dict(zip(packet["ids"], actions)))
            while result["kind"] == "need_actions":
                newborn = result["packet"]
                masks = [result["spawn_mask"]] * len(newborn["ids"])
                actions, _ = execution.act(newborn["observations"], spawn_mask=masks)
                result = env.continue_with(dict(zip(newborn["ids"], actions)))
            packet = result["packet"]
            if result["done"]:
                complete = True
                if viewer:
                    viewer.draw(env.core.env)
                break
    finally:
        policy.train(training)
        if viewer:
            viewer.close()
    return dict(env.metrics.report(env.state, complete=complete),
                policy_mode=policy_mode, policy_seed=execution.seed)


def validate(policy, config, horizon, deadline=float("inf")):
    # Keep validation simulation in a CPU worker so the trainer can enforce its
    # remaining wall budget even during expensive world initialization.
    from dataclasses import replace
    from rl.workers import WorkerPool
    from rl.execution import PolicyExecution
    from rl.validation import preserve_training_rng, validation_plan
    episodes = []
    if time.monotonic() >= deadline:
        return episodes
    training = policy.training
    with preserve_training_rng():
        policy.eval()
        try:
            with WorkerPool(replace(config, workers=1)) as pool:
                pool.deadline = deadline
                for item in validation_plan(config):
                    if time.monotonic() >= deadline:
                        break
                    execution = PolicyExecution(policy, item["policy_mode"], item["policy_seed"])
                    packet = pool.exchange({0: ("reset", dict(seed=item["environment_seed"], horizon=horizon))})[0]
                    while time.monotonic() < deadline:
                        actions, _ = execution.act(packet["observations"])
                        response = pool.exchange({0: ("begin", dict(zip(packet["ids"], actions)))})[0]
                        while response["kind"] == "need_actions":
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Validation deadline reached")
                            newborn = response["packet"]
                            actions, _ = execution.act(newborn["observations"],
                                spawn_mask=[response["spawn_mask"]] * len(newborn["ids"]))
                            response = pool.exchange({0: ("continue", dict(zip(newborn["ids"], actions)))})[0]
                        packet = response["packet"]
                        if response["done"]:
                            episodes.append(dict(response["episode"], **item))
                            break
        except TimeoutError:
            if time.monotonic() < deadline:
                raise
        finally:
            policy.train(training)
    return episodes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--horizon", type=float, default=3000.)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("rl/runs/evaluation.csv"))
    parser.add_argument("--policy-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--policy-rollouts", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 < args.horizon <= 3000:
        parser.error("horizon must be in (0, 3000]")
    if args.policy_rollouts < 1:
        parser.error("policy-rollouts must be positive")
    if args.policy_mode == "stochastic" and not 0 <= args.policy_seed <= 2**63 - args.policy_rollouts:
        parser.error("Derived policy seeds must stay in [0, 2**63-1]")
    return args


def prepare_output(path):
    """Migrate old deterministic-only evaluation CSVs without dropping metrics."""
    if not path.exists() or not path.stat().st_size:
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if "policy_mode" in fields and "policy_seed" in fields:
            return
        if "policy_mode" in fields or "policy_seed" in fields or not {"seed", "official_score"} <= set(fields):
            raise ValueError(f"Unrecognized evaluation CSV schema: {path}")
        rows = list(reader)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["policy_mode", "policy_seed"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, policy_mode="deterministic", policy_seed=""))
    temporary.replace(path)


def aggregate(episodes):
    complete = [e for e in episodes if e["complete"]]
    if not complete:
        return dict(completed_episodes=0)
    from rl.metrics import percentile25
    survival = [e["survival_time"] for e in complete]
    return dict(completed_episodes=len(complete), mean_survival=statistics.mean(survival),
                median_survival=statistics.median(survival), q25_survival=percentile25(survival),
                median_official_score=statistics.median(e["official_score"] for e in complete), min_survival=min(survival),
                max_survival=max(survival), mean_official_score=statistics.mean(e["official_score"] for e in complete),
                horizon_completion_rate=statistics.mean(e["horizon_reached"] for e in complete))


def main(argv=None):
    args = parse_args(argv)
    import torch
    from rl.config import Config
    from rl.checkpoint import load_checkpoint
    from rl.policy import SharedPolicy, choose_device
    from rl.metrics import CSVLogger, validation_rank
    from rl.execution import rollout_seeds
    torch.set_num_threads(1)
    saved = load_checkpoint(args.checkpoint)
    config = Config(**saved["config"])
    seeds = args.seeds or list(config.validation_seeds)
    if args.render and len(seeds) != 1:
        raise ValueError("Rendering requires exactly one --seeds value")
    policy = SharedPolicy(config.hidden_sizes).to(choose_device(args.device))
    load_checkpoint(args.checkpoint, policy)
    log = CSVLogger(args.output)
    prepare_output(args.output)
    episodes = []
    for seed, policy_seed in rollout_seeds(seeds, args.policy_mode, args.policy_seed, args.policy_rollouts):
        episode = evaluate_episode(policy, config, seed, args.horizon, render=args.render,
                                   policy_mode=args.policy_mode, policy_seed=policy_seed)
        log.write(episode)
        episodes.append(episode)
        print(episode, flush=True)
        if not episode["complete"]:
            break
    print("Validation rank:", validation_rank(episodes, args.policy_mode, args.policy_rollouts))
    print("Aggregate (completed episodes only):", aggregate(episodes))


if __name__ == "__main__":
    main()
