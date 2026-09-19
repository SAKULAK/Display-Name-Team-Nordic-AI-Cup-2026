"""Baseline hyperparameters with an isolated, fixed-reward experiment contract."""
from dataclasses import dataclass
from pathlib import Path
from rl.config import Config as BaselineConfig, Curriculum, worker_seed, VALIDATION_SEEDS


@dataclass
class Config(BaselineConfig):
    workers: int = 6
    rollout_steps: int = 128
    validation_seconds: float = 3600.
    validation_policy_mode: str = "stochastic"
    validation_policy_rollouts: int = 3
    validation_policy_seed_base: int = 20260918
    max_hours: float = 8.
    output_dir: str = "rl_hier/runs/pc_overnight"

    def __post_init__(self):
        super().__post_init__()
        if (self.energy_shaping, self.extinction_penalty, self.horizon_bonus) != (0., 5., 5.):
            raise ValueError("Hierarchical reward is fixed: score delta -5 extinction / +5 completion")
        if self.horizons != (300., 900., 1800., 3000.):
            raise ValueError("Use the baseline natural horizon curriculum")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        protect_output(self.output_dir)


def protect_output(path):
    """Never permit a run to write into the protected baseline or simulator."""
    root = Path(__file__).resolve().parents[1]
    destination = Path(path).resolve()
    for name in ("rl", "src"):
        protected = root / name
        if destination == protected or protected in destination.parents:
            raise ValueError(f"Output cannot be inside protected {name}/")
