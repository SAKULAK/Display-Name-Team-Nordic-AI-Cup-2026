"""Shared-policy PPO entrypoint. Training runs only when explicitly invoked."""
import argparse
from pathlib import Path
import random
import signal
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for Config")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init-from", type=Path, help="Initialize actor/critic weights only; start a fresh experiment")
    parser.add_argument("--start-horizon", type=float, help="Initial horizon value from config.horizons (e.g. 3000); not for resume")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--validation-seconds", type=float)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--rollout-steps", type=int)
    args = parser.parse_args(argv)
    if args.resume and args.start_horizon is not None:
        parser.error("--start-horizon cannot be combined with --resume; resume restores its curriculum")
    return args


def initialize_training(args):
    """Construct learning state only; never creates workers or runs training."""
    import json
    import numpy as np
    import torch
    from rl.config import Config, Curriculum
    from rl.policy import SharedPolicy, choose_device
    from rl.checkpoint import load_checkpoint

    payload = load_checkpoint(args.resume) if args.resume else None
    settings = dict(payload["config"]) if payload else {}
    if args.config:
        settings.update(json.loads(args.config.read_text(encoding="utf-8")))
    for name in ("workers", "max_hours", "device", "output_dir", "rollout_steps", "learning_rate", "base_seed", "validation_seconds"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    config = Config(**settings)
    if args.start_horizon is not None and args.start_horizon not in config.horizons:
        raise ValueError(f"--start-horizon must be one of {config.horizons}")
    if args.init_from:
        output = Path(config.output_dir).resolve()
        if output == args.init_from.resolve().parent or output.exists() and any(output.iterdir()):
            raise ValueError("--init-from requires a new empty output directory, separate from the source checkpoint")
    if payload and config.workers != payload["config"]["workers"]:
        raise ValueError("Resume must keep worker count to preserve disjoint per-worker seed streams")
    if payload and config.base_seed != payload["config"]["base_seed"]:
        raise ValueError("Resume must keep base_seed")
    torch.set_num_threads(1)
    random.seed(config.base_seed)
    np.random.seed(config.base_seed)
    torch.manual_seed(config.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.base_seed)
    policy = SharedPolicy(config.hidden_sizes).to(choose_device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    curriculum = Curriculum(config)
    if args.start_horizon is not None:
        curriculum.stage = config.horizons.index(args.start_horizon)
    state = dict(training_step=0, environment_ticks=0, decision_steps=0, episodes_completed=0,
                 births=0, deaths=0, population_max=0, wall_time=0., best_ranks={},
                 episode_indices=[0] * config.workers)
    if payload:
        load_checkpoint(args.resume, policy, optimizer, restore_rng=True)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
        state.update(payload["state"])
        curriculum.load_state_dict(state["curriculum"])
    elif args.init_from:
        from rl.checkpoint import initialize_weights
        state.update(initialize_weights(args.init_from, policy))
    return config, policy, optimizer, curriculum, state


def main(argv=None):
    args = parse_args(argv)
    # Kept inside main: Windows spawn workers must not import torch/create GPUs.
    import json
    import numpy as np
    import torch
    from rl.config import Config, Curriculum
    from rl.policy import SharedPolicy, choose_device
    from rl.checkpoint import load_checkpoint, save_checkpoint, save_periodic
    from rl.workers import WorkerPool
    from rl.rollout import Collector, prepare_batch
    from rl.ppo import update
    from rl.metrics import CSVLogger
    from rl.validation import process_validation
    from rl.evaluate import validate

    config, policy, optimizer, curriculum, state = initialize_training(args)
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    training_log = CSVLogger(directory / "training.csv")
    episode_log = CSVLogger(directory / "episodes.csv")
    tensorboard = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard = SummaryWriter(str(directory / "tensorboard"))
    except ImportError:
        pass
    started = time.monotonic()
    previous_wall = state["wall_time"]
    deadline = started + config.max_hours * 3600 - min(config.shutdown_margin_seconds, config.max_hours * 360)
    last_checkpoint = last_validation = last_stage_time = started
    collector = pool = None

    def snapshot():
        state["wall_time"] = previous_wall + time.monotonic() - started
        state["curriculum"] = curriculum.state_dict()
        state["curriculum_stage"] = curriculum.stage
        state["environment_steps"] = state["environment_ticks"]
        if collector is not None:
            state["episode_indices"] = list(collector.episode_indices)
        return state.copy()

    def stop_signal(*_):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_signal)
    print(f"Device={policy.device}; workers={config.workers}; horizon={curriculum.horizon}; output={directory}", flush=True)
    try:
        pool = WorkerPool(config)
        pool.deadline = deadline
        collector = Collector(pool, policy, config, state["episode_indices"])
        collector.reset(range(config.workers), curriculum.horizon)
        while time.monotonic() < deadline:
            transitions, episodes = [], []
            update_started = time.monotonic()
            for _ in range(config.rollout_steps):
                if time.monotonic() >= deadline:
                    break
                current, finished = collector.step(curriculum.horizon)
                transitions.extend(current)
                episodes.extend(finished)
                state["environment_ticks"] += sum(t["ticks"] for t in current)
                state["decision_steps"] += len(current)
                state["episodes_completed"] += len(finished)
                state["births"] += sum(t["births"] for t in current)
                state["deaths"] += sum(t["deaths"] for t in current)
                state["population_max"] = max(state["population_max"], *(t["peak_population"] for t in current))
                for episode in finished:
                    episode_log.write(episode)
                now = time.monotonic()
                if curriculum.advance(len(current), now - last_stage_time, finished):
                    print(f"Promoted to horizon {curriculum.horizon}; applies on each worker's next reset", flush=True)
                last_stage_time = now
                if now - last_checkpoint >= config.checkpoint_seconds:
                    save_periodic(directory, policy, optimizer, config, snapshot())
                    last_checkpoint = time.monotonic()
            if not transitions:
                break
            batch = prepare_batch(transitions, collector.values(), config)
            losses = update(policy, optimizer, batch, config, deadline)
            state["training_step"] += 1
            now = time.monotonic()
            energies = np.concatenate([p["observations"][:, 0] * 1000 for p in collector.packets.values()])
            populations = [len(p["ids"]) for p in collector.packets.values()]
            row = dict(wall_time=previous_wall + now - started, environment_ticks=state["environment_ticks"],
                       decision_steps=state["decision_steps"], training_step=state["training_step"],
                       ticks_per_second=sum(t["ticks"] for t in transitions) / max(now - update_started, 1e-9),
                       episodes_completed=state["episodes_completed"],
                       training_reward=float(np.mean([t["reward"] for t in transitions])),
                       official_reward=float(np.mean([t["official_reward"] for t in transitions])),
                       survival_time=float(np.mean([e["survival_time"] for e in episodes])) if episodes else None,
                       population_mean=float(np.mean(populations)), population_max=state["population_max"],
                       birth_count=state["births"], death_count=state["deaths"],
                       mean_energy=float(np.mean(energies)) if len(energies) else 0.,
                       median_energy=float(np.median(energies)) if len(energies) else 0.,
                       curriculum_stage=curriculum.stage, horizon=curriculum.horizon, **losses)
            training_log.write(row)
            if tensorboard:
                for key, value in row.items():
                    if value is not None:
                        tensorboard.add_scalar(key, value, state["decision_steps"])
            print(f"update={state['training_step']} ticks={state['environment_ticks']} "
                  f"ticks/s={row['ticks_per_second']:.1f} stage={curriculum.stage} loss={losses['actor_loss']:.4f}", flush=True)
            if now - last_checkpoint >= config.checkpoint_seconds:
                save_periodic(directory, policy, optimizer, config, snapshot())
                last_checkpoint = time.monotonic()
            if now - last_validation >= config.validation_seconds and now < deadline:
                validation = validate(policy, config, curriculum.horizon, deadline)
                process_validation(validation, config, state, curriculum.stage, curriculum.horizon, directory,
                                   lambda name: save_checkpoint(directory / name, policy, optimizer, config, snapshot()))
                last_validation = time.monotonic()
    except KeyboardInterrupt:
        print("Stopping; saving final checkpoint...", flush=True)
    except TimeoutError:
        if time.monotonic() < deadline:
            raise
        print("Wall-clock budget reached; saving final checkpoint...", flush=True)
    finally:
        try:
            save_periodic(directory, policy, optimizer, config, snapshot())
        finally:
            if pool is not None:
                pool.close()
            if tensorboard:
                tensorboard.close()
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
