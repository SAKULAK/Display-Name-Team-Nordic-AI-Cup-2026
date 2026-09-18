"""Serializable experiment configuration, seed partition, and horizon curriculum."""
from collections import deque
from dataclasses import asdict, dataclass, field

TRAIN_SEED_LIMIT = 2**31
VALIDATION_SEEDS = tuple(TRAIN_SEED_LIMIT + i for i in range(5))


@dataclass
class Config:
    base_seed: int = 17
    workers: int = 4
    hidden_sizes: tuple = (256, 256)
    gamma: float = .9995
    gae_lambda: float = .95
    clip: float = .2
    learning_rate: float = 3e-4
    epochs: int = 6
    minibatch_size: int = 1024
    rollout_steps: int = 128
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5
    target_kl: float = .03
    dt: float = .1
    action_repeat: int = 5
    extinction_penalty: float = 5.
    horizon_bonus: float = 5.
    energy_shaping: float = 0.
    horizons: tuple = (300., 900., 1800., 3000.)
    curriculum_window: int = 20
    curriculum_success_rate: float = .7
    curriculum_max_decisions: int = 200_000
    curriculum_max_seconds: float = 5400.
    validation_seeds: tuple = field(default_factory=lambda: VALIDATION_SEEDS)
    validation_policy_mode: str = "deterministic"
    validation_policy_rollouts: int = 1
    validation_policy_seed_base: int = 0
    checkpoint_seconds: float = 900.
    validation_seconds: float = 1800.
    max_hours: float = 6.
    shutdown_margin_seconds: float = 30.
    worker_timeout_seconds: float = 180.
    device: str = "auto"
    output_dir: str = "rl/runs/experiment"

    def __post_init__(self):
        self.hidden_sizes = tuple(self.hidden_sizes)
        self.horizons = tuple(self.horizons)
        self.validation_seeds = tuple(self.validation_seeds)
        if self.dt != .1 or self.action_repeat != 5:
            raise ValueError("This experiment uses the official dt=0.1 and five-tick decisions")
        if self.workers < 1 or self.rollout_steps < 1 or self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("workers, rollout_steps, epochs and minibatch_size must be positive")
        if not 0 <= self.base_seed < TRAIN_SEED_LIMIT:
            raise ValueError("Training base seed must be in [0, 2**31)")
        if not self.validation_seeds or any(s < TRAIN_SEED_LIMIT for s in self.validation_seeds):
            raise ValueError("Validation seeds must be reserved seeds >= 2**31")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("Validation seeds must be unique")
        if self.validation_policy_mode not in ("deterministic", "stochastic"):
            raise ValueError("validation_policy_mode must be deterministic or stochastic")
        if type(self.validation_policy_rollouts) is not int or self.validation_policy_rollouts < 1:
            raise ValueError("validation_policy_rollouts must be a positive integer")
        if self.validation_policy_mode == "deterministic" and self.validation_policy_rollouts != 1:
            raise ValueError("Deterministic validation requires validation_policy_rollouts=1; repeats are redundant")
        count = len(self.validation_seeds) * self.validation_policy_rollouts
        if (type(self.validation_policy_seed_base) is not int
                or not 0 <= self.validation_policy_seed_base <= 2**63 - count):
            raise ValueError("Derived validation policy seeds must stay in [0, 2**63-1]")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid discounts")
        if not self.horizons or any(h <= 0 or h > 3000 for h in self.horizons):
            raise ValueError("Horizons must lie in (0, 3000]")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Curriculum horizons must be strictly increasing")
        if not 0 <= self.energy_shaping <= .1:
            raise ValueError("Energy shaping must stay small, in [0, 0.1]")
        if min(self.max_hours, self.checkpoint_seconds, self.validation_seconds,
               self.worker_timeout_seconds, self.curriculum_max_seconds) <= 0:
            raise ValueError("Time limits must be positive")

    def to_dict(self):
        return asdict(self)


def worker_seed(base_seed, worker_id, episode_index, workers):
    if not 0 <= worker_id < workers or episode_index < 0:
        raise ValueError("Invalid worker/episode index")
    seed = base_seed + episode_index * workers + worker_id
    if not 0 <= seed < TRAIN_SEED_LIMIT:
        raise ValueError("Training seed stream exhausted its disjoint seed partition")
    return seed


class Curriculum:
    def __init__(self, config):
        self.config = config
        self.stage = 0
        self.stage_decisions = 0
        self.stage_seconds = 0.
        self.recent = deque(maxlen=config.curriculum_window)

    @property
    def horizon(self):
        return self.config.horizons[self.stage]

    def advance(self, decisions=0, seconds=0., episodes=()):
        self.stage_decisions += decisions
        self.stage_seconds += seconds
        for episode in episodes:
            if episode["horizon"] == self.horizon:
                self.recent.append(bool(episode["horizon_reached"]))
        successful = (len(self.recent) >= self.config.curriculum_window
                      and sum(self.recent) / len(self.recent) >= self.config.curriculum_success_rate)
        expired = (self.stage_decisions >= self.config.curriculum_max_decisions
                   or self.stage_seconds >= self.config.curriculum_max_seconds)
        if self.stage + 1 < len(self.config.horizons) and (successful or expired):
            self.stage += 1
            self.stage_decisions = 0
            self.stage_seconds = 0.
            self.recent.clear()
            return True
        return False

    def state_dict(self):
        return dict(stage=self.stage, stage_decisions=self.stage_decisions,
                    stage_seconds=self.stage_seconds, recent=list(self.recent))

    def load_state_dict(self, state):
        self.stage = state["stage"]
        self.stage_decisions = state["stage_decisions"]
        self.stage_seconds = state["stage_seconds"]
        self.recent = deque(state["recent"], maxlen=self.config.curriculum_window)
