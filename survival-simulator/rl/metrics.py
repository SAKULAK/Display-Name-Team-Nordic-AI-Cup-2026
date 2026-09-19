"""Reporting only; diagnostic values never feed actions or shaped rewards."""
import csv
import math
from pathlib import Path
import statistics
import time


class EpisodeMetrics:
    def __init__(self, seed, horizon):
        self.seed, self.horizon = seed, horizon
        self.started = time.monotonic()
        self.births = self.deaths = 0
        self.populations = []
        self.energies = []
        self.official_reward = self.training_reward = 0.
        self.ticks = self.decisions = 0

    def observe(self, agents, births=0, deaths=0):
        self.births += births
        self.deaths += deaths
        self.populations.append(len(agents))
        self.energies.extend(a["energy"] for a in agents)

    def report(self, state, complete=True):
        population = len(state["observations"])
        return dict(seed=self.seed, horizon=self.horizon, complete=int(complete),
                    official_score=state["score"], survival_time=state["sim_time"],
                    horizon_reached=int(population > 0 and state["sim_time"] + 1e-8 >= self.horizon),
                    final_population=population, peak_population=max(self.populations, default=0),
                    minimum_population=min(self.populations, default=0),
                    mean_population=statistics.mean(self.populations) if self.populations else 0.,
                    births=self.births, deaths=self.deaths,
                    predator_related_deaths=None, fruit_score_contribution=None,
                    mean_energy=statistics.mean(self.energies) if self.energies else 0.,
                    median_energy=statistics.median(self.energies) if self.energies else 0.,
                    official_reward=self.official_reward, training_reward=self.training_reward,
                    environment_ticks=self.ticks, decision_steps=self.decisions,
                    episode_wall_time=time.monotonic() - self.started)


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row):
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream))
            if header != list(row):
                raise ValueError(f"CSV schema changed: use a fresh log path: {self.path}")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def percentile25(values):
    """Linear interpolation at zero-based index (n-1)*0.25 (NumPy 'linear')."""
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    position = (len(values) - 1) * .25
    lower, upper = math.floor(position), math.ceil(position)
    return float(values[lower] + (position - lower) * (values[upper] - values[lower]))


def validation_statistics(episodes):
    if not episodes or any(not e["complete"] for e in episodes):
        return None
    survival = [e["survival_time"] for e in episodes]
    scores = [e["official_score"] for e in episodes]
    return dict(episode_count=len(episodes),
                completion_rate=statistics.mean(e["horizon_reached"] for e in episodes),
                mean_survival=statistics.mean(survival), median_survival=statistics.median(survival),
                q25_survival=percentile25(survival), min_survival=min(survival), max_survival=max(survival),
                mean_official_score=statistics.mean(scores), median_official_score=statistics.median(scores))


def validation_rank(episodes, policy_mode="deterministic", policy_rollouts=1):
    complete = [e for e in episodes if e["complete"]]
    if len(complete) != len(episodes) or not complete:
        return None
    if policy_mode == "stochastic" and policy_rollouts > 1:
        summary = validation_statistics(episodes)
        return (summary["completion_rate"], summary["q25_survival"],
                summary["median_survival"], summary["mean_official_score"])
    return (statistics.mean(e["horizon_reached"] for e in complete),
            statistics.median(e["survival_time"] for e in complete),
            statistics.mean(e["official_score"] for e in complete))
