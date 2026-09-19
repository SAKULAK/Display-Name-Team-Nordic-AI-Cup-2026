"""Baseline public actor input with Categorical + Beta + Bernoulli actions."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Bernoulli, Categorical
from rl.observation import OBS_DIM, GLOBAL_DIM
from rl.policy import mlp, choose_device
from rl_hier.primitives import MODE_NAMES


class SharedPolicy(nn.Module):
    def __init__(self, hidden_sizes=(256, 256)):
        super().__init__()
        self.actor = mlp(OBS_DIM, hidden_sizes, len(MODE_NAMES) + 5)
        self.critic = mlp(GLOBAL_DIM, hidden_sizes, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    @property
    def device(self):
        return next(self.parameters()).device

    def distributions(self, observations):
        if observations.shape[-1] != OBS_DIM:
            raise ValueError("Actor requires the baseline 103-element public observation")
        outputs = self.actor(observations)
        n = len(MODE_NAMES)
        concentrations = torch.nn.functional.softplus(outputs[:, n:n + 4]) + 1.
        return (Categorical(logits=outputs[:, :n]),
                Beta(concentrations[:, :2], concentrations[:, 2:]),
                Bernoulli(logits=outputs[:, -1]))

    def evaluate_actions(self, observations, actions, spawn_mask):
        mode, continuous, spawn = self.distributions(observations)
        # Store EXACT samples [mode, intensity, residual_unit, spawn]. The physical
        # residual is 2*u-1: include its affine Jacobian, without round-trip error.
        log_prob = mode.log_prob(actions[:, 0].long())
        log_prob += continuous.log_prob(actions[:, 1:3]).sum(-1) - math.log(2)
        log_prob += spawn.log_prob(actions[:, 3]) * spawn_mask
        entropy = mode.entropy() + continuous.entropy().sum(-1) + math.log(2)
        entropy += spawn.entropy() * spawn_mask
        return log_prob, entropy

    def value(self, contexts):
        return self.critic(contexts).squeeze(-1)

    @torch.no_grad()
    def act(self, observations, deterministic=False, spawn_mask=None):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if not len(observations):
            return np.empty((0, 4), np.float32), np.empty(0, np.float32)
        mode, continuous, spawn = self.distributions(observations)
        modes = mode.probs.argmax(-1) if deterministic else mode.sample()
        parameters = continuous.mean if deterministic else continuous.sample()
        binary = (spawn.probs >= .5).float() if deterministic else spawn.sample()
        mask = torch.ones_like(binary) if spawn_mask is None else torch.as_tensor(
            spawn_mask, dtype=torch.float32, device=self.device)
        actions = torch.cat((modes[:, None].float(), parameters, (binary * mask)[:, None]), -1)
        log_prob, _ = self.evaluate_actions(observations, actions, mask)
        return actions.cpu().numpy(), log_prob.cpu().numpy()
