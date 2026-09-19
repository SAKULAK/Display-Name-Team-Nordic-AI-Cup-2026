"""Population-normalized PPO updates; no simulator or behavior heuristics."""
import time
import numpy as np
import torch


def weighted_loss(values, weights, global_mean_weight):
    # Fixed rollout normalization, NOT a different denominator per minibatch.
    # Uniformly sampled minibatches are unbiased estimates of equal-team-step loss.
    return (values * weights).mean() / global_mean_weight


def update(policy, optimizer, batch, config, deadline=float("inf")):
    data = {k: torch.as_tensor(v, dtype=torch.float32, device=policy.device) for k, v in batch.items()}
    n = len(data["actions"])
    mean_weight = data["weights"].mean()
    totals = []
    policy.train()
    stop = False
    for _ in range(config.epochs):
        for indices in torch.randperm(n, device=policy.device).split(config.minibatch_size):
            if time.monotonic() >= deadline:
                stop = True
                break
            d = {k: v[indices] for k, v in data.items()}
            log_prob, entropy = policy.evaluate_actions(d["observations"], d["actions"], d["spawn_masks"])
            log_ratio = log_prob - d["log_probs"]
            ratio = log_ratio.exp()
            objective = torch.minimum(ratio * d["advantages"],
                                      ratio.clamp(1 - config.clip, 1 + config.clip) * d["advantages"])
            actor_loss = -weighted_loss(objective, d["weights"], mean_weight)
            value = policy.value(d["contexts"])
            clipped_value = d["old_values"] + (value - d["old_values"]).clamp(-config.clip, config.clip)
            value_error = torch.maximum((value - d["returns"]).square(), (clipped_value - d["returns"]).square())
            critic_loss = .5 * weighted_loss(value_error, d["weights"], mean_weight)
            entropy_bonus = weighted_loss(entropy, d["weights"], mean_weight)
            loss = actor_loss + config.value_coefficient * critic_loss - config.entropy_coefficient * entropy_bonus
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss; checkpoint preserved on exit")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                kl = weighted_loss((ratio - 1) - log_ratio, d["weights"], mean_weight)
                clipped = weighted_loss(((ratio - 1).abs() > config.clip).float(), d["weights"], mean_weight)
            totals.append([actor_loss.item(), critic_loss.item(), entropy_bonus.item(), kl.item(), clipped.item()])
            if kl.item() > config.target_kl:
                stop = True
                break
        if stop:
            break
    weights = batch["weights"].astype(np.float64)
    target, predicted = batch["returns"], batch["old_values"]
    target_mean = np.average(target, weights=weights)
    variance = np.average((target - target_mean) ** 2, weights=weights)
    error = target - predicted
    error_variance = np.average((error - np.average(error, weights=weights)) ** 2, weights=weights)
    explained = 1 - error_variance / variance if variance > 1e-12 else 0.
    averages = np.mean(totals, axis=0) if totals else np.zeros(5)
    return dict(zip(("actor_loss", "critic_loss", "entropy", "kl_estimate", "clip_fraction"), map(float, averages)),
                explained_variance=float(explained), optimizer_minibatches=len(totals))
