"""One shared actor, one team critic; Beta + Bernoulli bounded actions."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Bernoulli
from rl.observation import OBS_DIM, GLOBAL_DIM


def mlp(input_size, hidden_sizes, output_size):
    layers = []
    for size in hidden_sizes:
        layers.extend((nn.Linear(input_size, size), nn.Tanh()))
        input_size = size
    layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)


class SharedPolicy(nn.Module):
    def __init__(self, hidden_sizes=(256, 256)):
        super().__init__()
        self.actor = mlp(OBS_DIM, hidden_sizes, 7)
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
        outputs = self.actor(observations)
        # Concentrations >1 avoid boundary singularities; no Gaussian clipping.
        concentrations = torch.nn.functional.softplus(outputs[:, :6]) + 1.
        return Beta(concentrations[:, :3], concentrations[:, 3:]), Bernoulli(logits=outputs[:, 6])

    def evaluate_actions(self, observations, actions, spawn_mask):
        movement, spawn = self.distributions(observations)
        # Unit coordinates are stored. Two physical angles are affine transforms
        # with Jacobian 2*pi each; include their constant in density/entropy.
        jacobian = 2 * math.log(2 * math.pi)
        log_prob = movement.log_prob(actions[:, :3]).sum(-1) - jacobian
        log_prob += spawn.log_prob(actions[:, 3]) * spawn_mask
        entropy = movement.entropy().sum(-1) + jacobian + spawn.entropy() * spawn_mask
        return log_prob, entropy

    def value(self, contexts):
        return self.critic(contexts).squeeze(-1)

    @torch.no_grad()
    def act(self, observations, deterministic=False, spawn_mask=None):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if not len(observations):
            return np.empty((0, 4), np.float32), np.empty(0, np.float32)
        movement, spawn = self.distributions(observations)
        continuous = movement.mean if deterministic else movement.sample()
        binary = (spawn.probs >= .5).float() if deterministic else spawn.sample()
        mask = torch.ones_like(binary) if spawn_mask is None else torch.as_tensor(spawn_mask, dtype=torch.float32, device=self.device)
        # A newborn in a repeat cannot spawn until the next macro decision.
        binary = binary * mask
        actions = torch.cat((continuous, binary[:, None]), dim=-1)
        log_prob, _ = self.evaluate_actions(observations, actions, mask)
        return actions.cpu().numpy(), log_prob.cpu().numpy()


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
