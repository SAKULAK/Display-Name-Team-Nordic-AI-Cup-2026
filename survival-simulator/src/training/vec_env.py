"""
Subprocess-parallel vector wrapper around SurvivalEnv.

Environment stepping (physics, vectorized visibility raycasting) is pure
CPU-bound Python/NumPy work with zero GPU involvement, so running N_ENVS
copies sequentially in one process wastes every other core on the machine.
This runs each env in its own worker process and steps them concurrently:
the main process sends every worker its actions for the tick, then blocks on
all N_ENVS replies - the actual env.step() computation overlaps across
worker processes while we wait, instead of happening one env at a time.
"""
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np

from src.training.gym_env import SurvivalEnv


def _worker(remote, worker_env_remote):
    worker_env_remote.close()  # only the parent uses this end
    env = SurvivalEnv()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "set_difficulty":
            fruit_mult, tree_mult, predator_speed_mult = data
            env.set_difficulty(fruit_mult, tree_mult, predator_speed_mult)
            remote.send(None)
        elif cmd == "close":
            remote.close()
            break
        else:
            raise ValueError(f"unknown vec env command: {cmd}")


class SubprocVecSurvivalEnv:
    """Runs n_envs SurvivalEnv instances in separate worker processes."""

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        ctx = mp.get_context("spawn")
        pipes = [ctx.Pipe() for _ in range(n_envs)]
        self._remotes = [parent for parent, _ in pipes]
        worker_remotes = [child for _, child in pipes]

        self._processes = [
            ctx.Process(target=_worker, args=(worker_remote, remote), daemon=True)
            for remote, worker_remote in zip(self._remotes, worker_remotes)
        ]
        for p in self._processes:
            p.start()
        for worker_remote in worker_remotes:
            worker_remote.close()  # parent doesn't need the worker's end

    def reset(self) -> List[Dict[int, np.ndarray]]:
        for remote in self._remotes:
            remote.send(("reset", None))
        return [remote.recv()[0] for remote in self._remotes]  # drop the {} info half

    def reset_one(self, idx: int):
        """Reset a single env; returns the (obs, info) tuple SurvivalEnv.reset() would."""
        self._remotes[idx].send(("reset", None))
        return self._remotes[idx].recv()

    def reset_many(self, indices: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
        """Reset several envs, dispatching every reset command before waiting on any
        reply - like step(), so the actual work (full world regeneration: biome map,
        fruit/tree spawns - roughly 1000x a single step()'s cost, measured) overlaps
        across worker processes instead of paying each one back-to-back in the main
        process. Returns {idx: obs} (drops the {} info half, like reset())."""
        for idx in indices:
            self._remotes[idx].send(("reset", None))
        return {idx: self._remotes[idx].recv()[0] for idx in indices}

    def step(self, actions_per_env: List[Dict[int, Tuple[np.ndarray, float]]]):
        """
        Dispatch every env's actions for this tick, then wait for all replies - the
        actual env.step() work happens concurrently across worker processes during
        the recv loop below, not sequentially.
        """
        for remote, actions in zip(self._remotes, actions_per_env):
            remote.send(("step", actions))
        return [remote.recv() for remote in self._remotes]

    def set_difficulty(self, fruit_mult: float, tree_mult: float, predator_speed_mult: float = 1.0):
        for remote in self._remotes:
            remote.send(("set_difficulty", (fruit_mult, tree_mult, predator_speed_mult)))
        for remote in self._remotes:
            remote.recv()

    def close(self):
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
