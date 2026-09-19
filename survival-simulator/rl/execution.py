"""Evaluation-only execution modes; existing actor/distributions remain unchanged."""
import torch

POLICY_MODES = ("deterministic", "stochastic")
MAX_POLICY_SEED = 2**63 - 1


class PolicyExecution:
    """Isolated sampling stream for one episode, including newborn decisions.

    Distribution.sample() has no generator argument. Temporarily install private
    generator states only around actor calls, then restore ambient Torch states.
    Python/NumPy/simulator RNGs are never seeded or changed here. Use serially in
    one process (as the evaluator does), not concurrently from multiple threads.
    """
    def __init__(self, policy, mode="deterministic", seed=0):
        if mode not in POLICY_MODES:
            raise ValueError(f"Unknown policy mode: {mode}")
        self.policy, self.mode = policy, mode
        self.seed = seed if mode == "stochastic" else None
        self.cuda_index = None
        if mode == "stochastic":
            if not isinstance(seed, int) or not 0 <= seed <= MAX_POLICY_SEED:
                raise ValueError("policy seed must be an integer in [0, 2**63-1]")
            self.cpu_state = torch.Generator(device="cpu").manual_seed(seed).get_state()
            if policy.device.type == "cuda":
                self.cuda_index = policy.device.index
                if self.cuda_index is None:
                    self.cuda_index = torch.cuda.current_device()
                self.cuda_state = torch.Generator(device=f"cuda:{self.cuda_index}").manual_seed(seed).get_state()

    def act(self, observations, spawn_mask=None):
        if self.mode == "deterministic":
            return self.policy.act(observations, deterministic=True, spawn_mask=spawn_mask)
        devices = [] if self.cuda_index is None else [self.cuda_index]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_index is not None:
                torch.cuda.set_rng_state(self.cuda_state, self.cuda_index)
            try:
                return self.policy.act(observations, deterministic=False, spawn_mask=spawn_mask)
            finally:
                self.cpu_state = torch.get_rng_state()
                if self.cuda_index is not None:
                    self.cuda_state = torch.cuda.get_rng_state(self.cuda_index)


def rollout_seeds(environment_seeds, mode="deterministic", policy_seed=0, policy_rollouts=1):
    if mode not in POLICY_MODES or policy_rollouts < 1:
        raise ValueError("Invalid policy mode or rollout count")
    if mode == "stochastic" and not 0 <= policy_seed <= MAX_POLICY_SEED - (policy_rollouts - 1):
        raise ValueError("Derived policy seeds must stay in [0, 2**63-1]")
    for environment_seed in environment_seeds:
        for index in range(policy_rollouts):
            yield environment_seed, policy_seed + index if mode == "stochastic" else None
