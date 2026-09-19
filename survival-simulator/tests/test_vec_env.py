"""
Plumbing tests for SubprocVecSurvivalEnv - the multiprocessing wrapper that runs
each parallel SurvivalEnv in its own worker process (see vec_env.py's module
docstring for why: env stepping is CPU-bound with zero GPU involvement).
"""
import unittest

import numpy as np

from src.training.vec_env import SubprocVecSurvivalEnv


class SubprocVecSurvivalEnvTests(unittest.TestCase):
    def test_reset_returns_one_obs_dict_per_env(self):
        vec_env = SubprocVecSurvivalEnv(2)
        try:
            obs_list = vec_env.reset()
            self.assertEqual(len(obs_list), 2)
            for obs in obs_list:
                self.assertGreater(len(obs), 0)  # starting_agents defaults to 5
        finally:
            vec_env.close()

    def test_step_advances_all_envs_and_reports_population_level_flags(self):
        vec_env = SubprocVecSurvivalEnv(2)
        try:
            obs_list = vec_env.reset()
            for _ in range(5):
                actions_per_env = [
                    {aid: (np.array([0.2, 0.0, 1.0, 0.0], dtype=np.float32), 0.0) for aid in obs}
                    for obs in obs_list
                ]
                results = vec_env.step(actions_per_env)
                self.assertEqual(len(results), 2)
                obs_list = [r[0] for r in results]
                for next_obs, rewards, terminated, truncated, infos in results:
                    self.assertIn("__all__", terminated)
                    self.assertIn("__all__", truncated)
                    self.assertIn("__all__", infos)
        finally:
            vec_env.close()

    def test_set_difficulty_does_not_raise(self):
        vec_env = SubprocVecSurvivalEnv(1)
        try:
            vec_env.set_difficulty(1.5, 1.3)
        finally:
            vec_env.close()

    def test_reset_one_resets_only_that_env(self):
        vec_env = SubprocVecSurvivalEnv(2)
        try:
            vec_env.reset()
            obs, info = vec_env.reset_one(0)
            self.assertGreater(len(obs), 0)
            self.assertEqual(info, {})
        finally:
            vec_env.close()

    def test_reset_many_resets_every_requested_env(self):
        vec_env = SubprocVecSurvivalEnv(3)
        try:
            vec_env.reset()
            reset_obs = vec_env.reset_many([0, 2])
            self.assertEqual(set(reset_obs.keys()), {0, 2})
            for obs in reset_obs.values():
                self.assertGreater(len(obs), 0)
        finally:
            vec_env.close()

    def test_reset_many_with_empty_list_returns_empty_dict(self):
        vec_env = SubprocVecSurvivalEnv(2)
        try:
            vec_env.reset()
            self.assertEqual(vec_env.reset_many([]), {})
        finally:
            vec_env.close()


if __name__ == "__main__":
    unittest.main()
