"""Tests for train.py's checkpoint persistence and GAE computation."""
import os
import tempfile
import unittest

import numpy as np
import torch

import src.training.train as train_module
from src.training.gym_env import OBS_DIM
from src.training.model import ActorCritic
from src.training.train import GAMMA, GAE_LAMBDA, Trajectory, collect_rollout, compute_gae, load_checkpoint, save_checkpoint
from src.training.vec_env import SubprocVecSurvivalEnv


class CheckpointRoundTripTests(unittest.TestCase):
    def test_save_and_load_restores_identical_weights_and_update_counter(self):
        model = ActorCritic(obs_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # Take one real optimizer step so there's real optimizer state to round-trip.
        dummy_obs = torch.randn(4, 16)
        _, _, log_prob, _, value = model.get_action_and_value(dummy_obs)
        loss = -log_prob.mean() + value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        original_path = train_module.CHECKPOINT_PATH
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_module.CHECKPOINT_PATH = os.path.join(tmp_dir, "policy.pt")
            try:
                save_checkpoint(model, optimizer, update=42)

                fresh_model = ActorCritic(obs_dim=16)
                fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=3e-4)
                start_update = load_checkpoint(fresh_model, fresh_optimizer)

                self.assertEqual(start_update, 43)
                for (name, original_param), (_, loaded_param) in zip(
                    model.state_dict().items(), fresh_model.state_dict().items()
                ):
                    self.assertTrue(torch.equal(original_param, loaded_param), name)
            finally:
                train_module.CHECKPOINT_PATH = original_path

    def test_missing_checkpoint_starts_from_update_one(self):
        model = ActorCritic(obs_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        original_path = train_module.CHECKPOINT_PATH
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_module.CHECKPOINT_PATH = os.path.join(tmp_dir, "does_not_exist.pt")
            try:
                self.assertEqual(load_checkpoint(model, optimizer), 1)
            finally:
                train_module.CHECKPOINT_PATH = original_path


class GAETests(unittest.TestCase):
    def test_matches_hand_computed_two_step_trajectory(self):
        # rewards=[1.0, 1.0], values=[0.0, 0.0], bootstrap=0.0:
        #   delta_1 = 1.0 + GAMMA*0.0 - 0.0 = 1.0  ->  gae_1 = 1.0
        #   delta_0 = 1.0 + GAMMA*0.0 - 0.0 = 1.0  ->  gae_0 = 1.0 + GAMMA*GAE_LAMBDA*gae_1
        traj = Trajectory(reward=[1.0, 1.0], value=[0.0, 0.0])
        advantages, returns = compute_gae(traj, bootstrap_value=0.0)
        expected_gae_1 = 1.0
        expected_gae_0 = 1.0 + GAMMA * GAE_LAMBDA * expected_gae_1
        np.testing.assert_allclose(advantages, [expected_gae_0, expected_gae_1], atol=1e-6)
        np.testing.assert_allclose(returns, [expected_gae_0, expected_gae_1], atol=1e-6)  # values are 0

    def test_all_zero_gives_all_zero(self):
        traj = Trajectory(reward=[0.0, 0.0, 0.0], value=[0.0, 0.0, 0.0])
        advantages, returns = compute_gae(traj, bootstrap_value=0.0)
        np.testing.assert_allclose(advantages, [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(returns, [0.0, 0.0, 0.0], atol=1e-9)

    def test_single_step_matches_one_step_td_error(self):
        traj = Trajectory(reward=[5.0], value=[2.0])
        advantages, returns = compute_gae(traj, bootstrap_value=3.0)
        expected_delta = 5.0 + GAMMA * 3.0 - 2.0
        np.testing.assert_allclose(advantages, [expected_delta], atol=1e-6)
        np.testing.assert_allclose(returns, [expected_delta + 2.0], atol=1e-6)


class CollectRolloutTests(unittest.TestCase):
    """collect_rollout defers a died env's reset to a single batched call at the end
    of the rollout (see its own comment) instead of resetting inline as soon as an
    env dies - verify that still leaves every env correctly reset and populated by
    the time the rollout returns, not stuck idle from a mid-rollout death."""

    def test_no_env_left_empty_after_a_rollout_with_mid_rollout_deaths(self):
        n_envs = 3
        vec_env = SubprocVecSurvivalEnv(n_envs)
        try:
            model = ActorCritic(obs_dim=OBS_DIM)
            obs_list = vec_env.reset()
            # Untrained/near-random policy: population wipeout within a few hundred
            # ticks is the norm (verified separately), so 300 ticks reliably
            # produces at least one mid-rollout death for a 3-env batch.
            segments, episode_summaries, latest_info_per_env, obs_list = collect_rollout(
                vec_env, model, obs_list, num_ticks=300
            )
            self.assertEqual(len(obs_list), n_envs)
            for obs in obs_list:
                self.assertGreater(len(obs), 0)  # every env has live agents again, none left stalled
            self.assertGreater(len(episode_summaries), 0)  # sanity: this test scenario actually exercised a death
        finally:
            vec_env.close()


if __name__ == "__main__":
    unittest.main()
