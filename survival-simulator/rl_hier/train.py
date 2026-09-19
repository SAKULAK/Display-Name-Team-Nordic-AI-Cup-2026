"""Hierarchical PPO trainer. Training starts only through an explicit main call."""
import argparse
import json
from pathlib import Path
import random
import signal
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    for name, kind in (("workers", int), ("max-hours", float), ("device", str),
                       ("output-dir", str), ("learning-rate", float),
                       ("validation-seconds", float), ("rollout-steps", int), ("base-seed", int)):
        parser.add_argument("--" + name, type=kind)
    return parser.parse_args(argv)


def initialize_training(args):
    """Build learning state only, with no simulator creation or optimization."""
    import numpy as np
    import torch
    from rl_hier.config import Config, Curriculum
    from rl_hier.policy import SharedPolicy, choose_device
    from rl_hier.checkpoint import load_checkpoint
    payload = load_checkpoint(args.resume) if args.resume else None
    settings = dict(payload["config"]) if payload else {}
    if args.config:
        settings.update(json.loads(args.config.read_text(encoding="utf-8")))
    for name in ("workers", "max_hours", "device", "output_dir", "learning_rate",
                 "validation_seconds", "rollout_steps", "base_seed"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    config = Config(**settings)
    if payload:
        runtime = {"max_hours", "device", "output_dir", "learning_rate", "validation_seconds",
                   "checkpoint_seconds", "shutdown_margin_seconds", "worker_timeout_seconds",
                   "validation_policy_mode", "validation_policy_rollouts", "validation_policy_seed_base",
                   "validation_seeds"}
        saved_config = Config(**payload["config"]).to_dict()
        for name, value in config.to_dict().items():
            if name not in runtime and value != saved_config[name]:
                raise ValueError(f"Resume must preserve {name}")
    elif Path(config.output_dir).exists() and any(Path(config.output_dir).iterdir()):
        raise ValueError("Fresh training requires an empty output directory; use --resume for an existing run")
    torch.set_num_threads(1)
    random.seed(config.base_seed)
    np.random.seed(config.base_seed)
    torch.manual_seed(config.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.base_seed)
    policy = SharedPolicy(config.hidden_sizes).to(choose_device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    curriculum = Curriculum(config)
    state = dict(training_step=0, environment_ticks=0, decision_steps=0, episodes_completed=0,
                 births=0, deaths=0, population_max=0, wall_time=0., best_ranks={},
                 episode_indices=[0] * config.workers, worker_snapshots=None,
                 checkpoint_elapsed=0., validation_elapsed=0.)
    if payload:
        load_checkpoint(args.resume, policy, optimizer, restore_rng=True)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
        state.update(payload["state"])
        curriculum.load_state_dict(state["curriculum"])
        if state["decision_steps"] and not state["worker_snapshots"]:
            raise ValueError("Resume checkpoint is missing live worker episodes")
    return config, policy, optimizer, curriculum, state


def restore_collector(collector, state, horizon):
    snapshots = state.get("worker_snapshots")
    if snapshots:
        if set(snapshots) != set(range(collector.config.workers)):
            raise ValueError("Incomplete worker snapshot set")
        collector.packets = collector.pool.exchange({i: ("restore", blob) for i, blob in snapshots.items()})
    else:
        collector.reset(range(collector.config.workers), horizon)


def main(argv=None):
    # All torch imports stay out of Windows worker module startup.
    import numpy as np
    from rl_hier.checkpoint import save_checkpoint, save_periodic
    from rl_hier.workers import WorkerPool
    from rl_hier.rollout import Collector, prepare_batch
    from rl_hier.ppo import update
    from rl_hier.metrics import CSVLogger, action_statistics
    from rl_hier.validation import process_validation, preserve_training_rng
    from rl_hier.evaluate import validate
    args = parse_args(argv)
    config, policy, optimizer, curriculum, state = initialize_training(args)
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    training_log, episode_log, mode_log = (CSVLogger(directory / name) for name in
                                         ("training.csv", "episodes.csv", "modes_by_stage.csv"))
    tensorboard = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard = SummaryWriter(str(directory / "tensorboard"))
    except ImportError:
        pass
    started = time.monotonic()
    previous_wall = state["wall_time"]
    deadline = started + config.max_hours * 3600 - min(config.shutdown_margin_seconds, config.max_hours * 360)
    last_checkpoint = started - state["checkpoint_elapsed"]
    last_validation = started - state["validation_elapsed"]
    last_stage_time = started
    collector = pool = None
    stop_requested = False
    safe_boundary = True
    checkpoint_failed = False

    def stop_signal(*_):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; completing the current update boundary before checkpointing.", flush=True)

    def snapshot():
        nonlocal checkpoint_failed
        now = time.monotonic()
        state.update(wall_time=previous_wall + now - started, curriculum=curriculum.state_dict(),
                     curriculum_stage=curriculum.stage, environment_steps=state["environment_ticks"],
                     checkpoint_elapsed=now - last_checkpoint, validation_elapsed=now - last_validation)
        if collector is not None:
            state["episode_indices"] = list(collector.episode_indices)
            # One worker at a time bounds peak parent/IPC snapshot memory.
            blobs = {}
            old_deadline = pool.deadline
            pool.deadline = float("inf")
            try:
                for wid in range(config.workers):
                    blobs[wid] = pool.exchange({wid: ("snapshot", None)})[wid]
            except BaseException:
                checkpoint_failed = True
                raise
            finally:
                pool.deadline = old_deadline
            state["worker_snapshots"] = blobs
        return state.copy()

    def periodic():
        nonlocal last_checkpoint
        last_checkpoint = time.monotonic()
        save_periodic(directory, policy, optimizer, config, snapshot())

    previous_signals = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    print(f"Device={policy.device}; workers={config.workers}; horizon={curriculum.horizon}; output={directory}", flush=True)
    try:
        # Preserve a usable checkpoint even if initial environment creation fails.
        # On resume, this retains the loaded live worker snapshots.
        periodic()
        with preserve_training_rng():
            pool = WorkerPool(config)
        pool.deadline = deadline
        collector = Collector(pool, policy, config, state["episode_indices"])
        safe_boundary = False
        restore_collector(collector, state, curriculum.horizon)
        safe_boundary = True
        while time.monotonic() < deadline and not stop_requested:
            transitions, episodes = [], []
            update_started = time.monotonic()
            safe_boundary = False
            for _ in range(config.rollout_steps):
                if time.monotonic() >= deadline or stop_requested:
                    break
                horizons = {i: p["horizon"] for i, p in collector.packets.items()}
                current, finished = collector.step(curriculum.horizon)
                for transition in current:
                    transition["horizon"] = horizons[transition["worker"]]
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
                    print(f"Promoted to horizon {curriculum.horizon}; applies on next worker reset", flush=True)
                last_stage_time = now
            if not transitions:
                safe_boundary = True
                break
            batch = prepare_batch(transitions, collector.values(), config)
            diagnostics = action_statistics(policy, batch)
            for horizon in sorted({t["horizon"] for t in transitions}):
                # Exact horizon of the episode, including workers still finishing
                # the previous stage after a curriculum promotion.
                selected = np.concatenate([np.full(len(t["actors"]["actions"]), t["horizon"] == horizon)
                                           for t in transitions])
                stage_batch = {key: value[selected] for key, value in batch.items()}
                mode_log.write(dict(training_step=state["training_step"] + 1,
                                    curriculum_stage=config.horizons.index(horizon), horizon=horizon,
                                    **action_statistics(policy, stage_batch)))
            losses = update(policy, optimizer, batch, config, deadline)
            state["training_step"] += 1
            safe_boundary = True
            now = time.monotonic()
            populations = [len(p["ids"]) for p in collector.packets.values()]
            energies = np.concatenate([p["observations"][:, 0] * 1000 for p in collector.packets.values()])
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
                       curriculum_stage=curriculum.stage, horizon=curriculum.horizon, **losses, **diagnostics)
            training_log.write(row)
            if tensorboard:
                for key, value in row.items():
                    if value is not None:
                        tensorboard.add_scalar(key, value, state["decision_steps"])
            print(f"update={state['training_step']} ticks={state['environment_ticks']} "
                  f"ticks/s={row['ticks_per_second']:.1f} stage={curriculum.stage}", flush=True)
            if now - last_checkpoint >= config.checkpoint_seconds:
                periodic()
            if now - last_validation >= config.validation_seconds and now < deadline and not stop_requested:
                validation = validate(policy, config, curriculum.horizon, deadline,
                                      should_stop=lambda: stop_requested)
                last_validation = time.monotonic()
                process_validation(validation, config, state, curriculum.stage, curriculum.horizon, directory,
                                   lambda name: save_checkpoint(directory / name, policy, optimizer, config, snapshot()))
    except KeyboardInterrupt:
        print("Interrupted; preserving the last consistent checkpoint.", flush=True)
    except TimeoutError:
        if time.monotonic() < deadline:
            raise
        print("Wall-clock budget reached.", flush=True)
    finally:
        try:
            if safe_boundary and not checkpoint_failed:
                periodic()
            else:
                print("Unfinished collection/update: latest.pt remains at the last consistent boundary.", flush=True)
        finally:
            if pool is not None:
                pool.close()
            if tensorboard:
                tensorboard.close()
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
