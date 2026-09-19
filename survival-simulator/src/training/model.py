from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

# Bounds on the learned log-std so exploration noise can't collapse to ~0 (which
# would silently lock the policy into whatever behavior it has already settled on)
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic used for every agent (homogeneous parameter sharing:
    one network is applied independently to each living agent's observation).

    Continuous head -> 4 values, sampled from a Gaussian with a state-independent
    learned std (standard PPO trick):
        0: move_distance_frac, squashed/scaled to [0, sprint_speed] outside this module
        1: sin(move_direction)
        2: cos(move_direction)
        3: turn_angle (radians)
    Discrete head -> 1 Bernoulli logit for the spawn_agent decision.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.continuous_mean = nn.Linear(hidden_dim, 4)
        # State-independent log-std, initialized for modest exploration noise.
        self.continuous_log_std = nn.Parameter(torch.zeros(4) - 0.5)

        self.spawn_logit = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mean = self.continuous_mean(features)
        std = self.continuous_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        spawn_logit = self.spawn_logit(features).squeeze(-1)
        value = self.value_head(features).squeeze(-1)
        return mean, std, spawn_logit, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Sample an action for the given observations, or evaluate log-probs/entropy/value
        for a previously-sampled (continuous_action, spawn_action) pair (used by PPO when
        recomputing log-probs for stored rollout data).

        Returns:
            continuous_action: (batch, 4)
            spawn_action: (batch,) of 0./1.
            log_prob: (batch,) summed log-prob across both heads
            entropy: (batch,) summed entropy across both heads
            value: (batch,)
        """
        mean, std, spawn_logit, value = self.forward(obs)

        cont_dist = Normal(mean, std)
        spawn_dist = Bernoulli(logits=spawn_logit)

        if action is None:
            continuous_action = cont_dist.sample()
            spawn_action = spawn_dist.sample()
        else:
            continuous_action, spawn_action = action

        log_prob = cont_dist.log_prob(continuous_action).sum(-1) + spawn_dist.log_prob(spawn_action)
        entropy = cont_dist.entropy().sum(-1) + spawn_dist.entropy()

        return continuous_action, spawn_action, log_prob, entropy, value
