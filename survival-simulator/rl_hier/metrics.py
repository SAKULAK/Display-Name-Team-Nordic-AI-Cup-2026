"""Hierarchical diagnostics. No metric feeds the reward or primitive selection."""
import numpy as np
import torch
from rl.metrics import CSVLogger, validation_statistics
from rl_hier.primitives import MODE_NAMES


def validation_rank(episodes, policy_mode=None, policy_rollouts=None):
    summary = validation_statistics(episodes)
    if summary is None:
        return None
    return tuple(summary[k] for k in ("completion_rate", "q25_survival", "median_survival", "mean_official_score"))


@torch.no_grad()
def action_statistics(policy, batch):
    """Call before update: probabilities describe the actual collecting policy.

    Distribution diagnostics are minibatched to bound GPU memory. Report both
    raw per-agent frequencies and equal-team-step mode frequencies.
    """
    actions, weights = batch["actions"], batch["weights"]
    masks = batch["spawn_masks"]
    entropy_sum = spawn_sum = 0.
    for start in range(0, len(actions), 1024):
        obs = torch.as_tensor(batch["observations"][start:start + 1024],
                              dtype=torch.float32, device=policy.device)
        mode, _, spawn = policy.distributions(obs)
        entropy_sum += mode.entropy().sum().item()
        mask = torch.as_tensor(masks[start:start + 1024], device=policy.device)
        spawn_sum += (spawn.probs * mask).sum().item()
    row = dict(mode_entropy=entropy_sum / len(actions),
               spawn_probability=spawn_sum / max(float(masks.sum()), 1.),
               spawn_action_frequency=float(actions[:, 3].sum() / max(masks.sum(), 1.)))
    for index, name in enumerate(MODE_NAMES):
        selected = actions[:, 0] == index
        row[f"mode_{name.lower()}_fraction"] = float(selected.mean())
        row[f"mode_{name.lower()}_team_fraction"] = float(np.average(selected, weights=weights))
    for name, values in (("movement_intensity", actions[:, 1]), ("steering_residual", 2 * actions[:, 2] - 1)):
        for statistic, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            row[f"{name}_{statistic}"] = float(fn(values))
    return row
