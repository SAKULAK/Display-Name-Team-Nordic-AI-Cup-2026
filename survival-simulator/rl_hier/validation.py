"""Validation isolation, full-sweep eligibility, reporting, and best selection."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random

from rl.config import VALIDATION_SEEDS
from rl_hier.metrics import CSVLogger, validation_rank, validation_statistics


@contextmanager
def preserve_training_rng():
    """Protect the entire sweep, including worker startup and exceptional exits."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def validation_plan(config):
    for environment_index, seed in enumerate(config.validation_seeds):
        for rollout_index in range(config.validation_policy_rollouts):
            policy_seed = None
            if config.validation_policy_mode == "stochastic":
                policy_seed = (config.validation_policy_seed_base
                               + environment_index * config.validation_policy_rollouts + rollout_index)
            yield dict(environment_seed=seed, rollout_index=rollout_index,
                       policy_mode=config.validation_policy_mode, policy_seed=policy_seed)


def complete_sweep(episodes, config, horizon):
    """Exact plan coverage, not just a row count; duplicates cannot qualify."""
    plan = list(validation_plan(config))
    if len(episodes) != len(plan) or any(not e.get("complete") for e in episodes):
        return False
    expected = {(p["environment_seed"], p["rollout_index"], p["policy_mode"], p["policy_seed"]) for p in plan}
    actual = set()
    for e in episodes:
        seed = e.get("environment_seed", e.get("seed"))
        if e.get("seed") != seed or e.get("horizon") != horizon:
            return False
        actual.add((seed, e.get("rollout_index", 0), e.get("policy_mode", "deterministic"), e.get("policy_seed")))
    return actual == expected


def rank_key(config, horizon):
    # A hierarchical checkpoint always uses the four-part robust rank.
    if config.validation_policy_mode == "deterministic" and tuple(config.validation_seeds) == VALIDATION_SEEDS:
        return str(horizon)
    protocol = dict(mode=config.validation_policy_mode, rollouts=config.validation_policy_rollouts,
                    policy_seed_base=config.validation_policy_seed_base, environment_seeds=list(config.validation_seeds))
    return str(horizon) + "|" + json.dumps(protocol, sort_keys=True, separators=(",", ":"))


def append_episode(path, row):
    """Extend legacy validation.csv atomically; retain every historical column."""
    path = Path(path)
    logger = CSVLogger(path)
    if not path.exists() or not path.stat().st_size:
        logger.write(row)
        return
    with path.open(newline="", encoding="utf-8") as stream:
        fields = next(csv.reader(stream))
    additions = [name for name in row if name not in fields]
    if additions:
        new_fields = fields + additions
        temporary = path.with_suffix(path.suffix + ".tmp")
        with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(dest, fieldnames=new_fields)
            writer.writeheader()
            for old in csv.DictReader(source):
                defaults = dict(validation_step=old.get("training_step", ""),
                                environment_seed=old.get("seed", ""), rollout_index=0,
                                policy_mode="deterministic", policy_seed="")
                old.update({key: defaults.get(key, "") for key in additions})
                writer.writerow(old)
        temporary.replace(path)
        fields = new_fields
    logger.write({name: row.get(name, "") for name in fields})


def process_validation(episodes, config, state, curriculum_stage, horizon, directory, save_best):
    """Report a sweep and save winners via callback; never executes simulations."""
    directory = Path(directory)
    for episode in episodes:
        row = dict(training_step=state["training_step"], curriculum_stage=curriculum_stage, **episode)
        row.update(validation_step=state["training_step"], environment_seed=episode["seed"],
                   rollout_index=episode.get("rollout_index", 0), policy_mode=episode.get("policy_mode", "deterministic"),
                   policy_seed=episode.get("policy_seed"))
        append_episode(directory / "validation.csv", row)
    expected = len(config.validation_seeds) * config.validation_policy_rollouts
    prefix = (f"Validation: mode={config.validation_policy_mode} seeds={len(config.validation_seeds)} "
              f"rollouts={config.validation_policy_rollouts} episodes={len(episodes)}/{expected} horizon={horizon}")
    if not complete_sweep(episodes, config, horizon):
        print(prefix + " incomplete/skipped; not eligible for best checkpoint", flush=True)
        return False
    summary = validation_statistics(episodes)
    rank = validation_rank(episodes, config.validation_policy_mode, config.validation_policy_rollouts)
    context = dict(training_step=state["training_step"], environment_ticks=state["environment_ticks"],
                   curriculum_stage=curriculum_stage, horizon=horizon, policy_mode=config.validation_policy_mode,
                   policy_rollouts=config.validation_policy_rollouts, num_environment_seeds=len(config.validation_seeds),
                   validation_policy_seed_base=config.validation_policy_seed_base)
    CSVLogger(directory / "validation_summary.csv").write(dict(context, **summary, validation_rank=json.dumps(rank)))
    seed_summaries = []
    for seed in config.validation_seeds:
        stats = validation_statistics([e for e in episodes if e["seed"] == seed])
        seed_summaries.append(f"{seed}:{stats['median_survival']:.1f}")
        CSVLogger(directory / "validation_by_seed.csv").write(dict(context, environment_seed=seed, **stats))
    print(prefix + f" completion_rate={summary['completion_rate']:.3f} q25_survival={summary['q25_survival']:.1f} "
          f"median_survival={summary['median_survival']:.1f} mean_survival={summary['mean_survival']:.1f} "
          f"min={summary['min_survival']:.1f} max={summary['max_survival']:.1f} "
          f"mean_score={summary['mean_official_score']:.3f} rank={rank}", flush=True)
    print("Validation seed medians: " + " ".join(seed_summaries), flush=True)
    key = rank_key(config, horizon)
    best = state["best_ranks"].get(key)
    if best is not None and len(best) == len(rank) and rank <= tuple(best):
        return False
    state["best_ranks"][key] = rank
    state.update(validation_policy_mode=config.validation_policy_mode,
                 validation_policy_rollouts=config.validation_policy_rollouts,
                 validation_policy_seed_base=config.validation_policy_seed_base,
                 validation_completion_rate=summary["completion_rate"],
                 validation_q25_survival=summary["q25_survival"],
                 validation_median_survival=summary["median_survival"],
                 validation_mean_survival=summary["mean_survival"],
                 validation_mean_official_score=summary["mean_official_score"], validation_rank=rank)
    save_best("best_validation.pt")
    save_best(f"best_validation_stage{curriculum_stage}.pt")
    return True
