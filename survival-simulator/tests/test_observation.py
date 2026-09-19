"""Tests for encode_observation's shape, value ranges, and presence-flag semantics."""
import unittest

import numpy as np

from src.training.gym_env import (
    AGENT_SLOT_DIM, BIOME_TYPES, EDGE_SLOT_DIM, FRUIT_SLOT_DIM, K_AGENT, K_EDGE,
    K_FRUIT, K_PREDATOR, K_TREE, OBS_DIM, OWN_STATE_DIM, PREDATOR_SLOT_DIM,
    TREE_SLOT_DIM, SurvivalEnv, encode_observation,
)


def _make_status(observations, **overrides):
    status = dict(
        agent_id=0, energy=150.0, biome="forest", age=10.0, speed=10.0,
        sprint_speed=20.0, hearing_radius=50.0, vision_angle=np.pi / 3,
        vision_range=200.0, max_energy=500.0, observations=observations,
    )
    status.update(overrides)
    return status


class ObservationShapeTests(unittest.TestCase):
    def test_empty_observations_has_correct_shape_and_is_finite(self):
        vec = encode_observation(_make_status([]), sim_time_frac=0.0)
        self.assertEqual(vec.shape, (OBS_DIM,))
        self.assertTrue(np.isfinite(vec).all())

    def test_overfull_observations_still_has_correct_shape_and_is_finite(self):
        # More entities than any bucket can hold, of every type - exercises the
        # sort-and-truncate path, not just the zero-padding path.
        observations = []
        for i in range(K_FRUIT + 3):
            observations.append(dict(type="Fruit", distance=10.0 + i, angle=0.1 * i))
        for i in range(K_TREE + 3):
            observations.append(dict(type="Tree", distance=10.0 + i, angle=0.1 * i))
        for i in range(K_PREDATOR + 3):
            observations.append(dict(type="Predator", distance=10.0 + i, angle=0.1 * i, rel_dir=0.2 * i))
        for i in range(K_AGENT + 3):
            observations.append(dict(type="Agent", distance=10.0 + i, angle=0.1 * i, rel_dir=0.2 * i))
        for i in range(K_EDGE + 3):
            observations.append(dict(type="Edge", coords=((10.0 + i, 0.0), (20.0 + i, 5.0))))
        vec = encode_observation(_make_status(observations), sim_time_frac=0.5)
        self.assertEqual(vec.shape, (OBS_DIM,))
        self.assertTrue(np.isfinite(vec).all())

    def test_unknown_biome_gives_all_zero_onehot(self):
        vec = encode_observation(_make_status([], biome="lava"), sim_time_frac=0.0)
        biome_onehot = vec[OWN_STATE_DIM:OWN_STATE_DIM + len(BIOME_TYPES)]
        self.assertTrue(np.all(biome_onehot == 0.0))

    def test_sim_time_frac_is_last_own_state_feature(self):
        vec_start = encode_observation(_make_status([]), sim_time_frac=0.0)
        vec_end = encode_observation(_make_status([]), sim_time_frac=1.0)
        self.assertAlmostEqual(float(vec_start[OWN_STATE_DIM - 1]), 0.0, places=6)
        self.assertAlmostEqual(float(vec_end[OWN_STATE_DIM - 1]), 1.0, places=6)


class PresenceFlagTests(unittest.TestCase):
    def test_empty_fruit_slots_have_presence_zero(self):
        vec = encode_observation(_make_status([]), sim_time_frac=0.0)
        offset = OWN_STATE_DIM + len(BIOME_TYPES)
        fruit_block = vec[offset: offset + K_FRUIT * FRUIT_SLOT_DIM].reshape(K_FRUIT, FRUIT_SLOT_DIM)
        self.assertTrue(np.all(fruit_block[:, -1] == 0.0))

    def test_one_real_fruit_marks_only_first_slot_present(self):
        observations = [dict(type="Fruit", distance=30.0, angle=0.5)]
        vec = encode_observation(_make_status(observations), sim_time_frac=0.0)
        offset = OWN_STATE_DIM + len(BIOME_TYPES)
        fruit_block = vec[offset: offset + K_FRUIT * FRUIT_SLOT_DIM].reshape(K_FRUIT, FRUIT_SLOT_DIM)
        presence = fruit_block[:, -1]
        self.assertEqual(presence[0], 1.0)
        self.assertTrue(np.all(presence[1:] == 0.0))


class AngleEncodingTests(unittest.TestCase):
    def test_fruit_angle_encoded_as_unit_sin_cos(self):
        observations = [dict(type="Fruit", distance=30.0, angle=2.5)]
        vec = encode_observation(_make_status(observations), sim_time_frac=0.0)
        offset = OWN_STATE_DIM + len(BIOME_TYPES)
        _, sin_a, cos_a, _ = vec[offset:offset + FRUIT_SLOT_DIM]
        self.assertAlmostEqual(float(sin_a), float(np.sin(2.5)), places=5)
        self.assertAlmostEqual(float(cos_a), float(np.cos(2.5)), places=5)
        self.assertAlmostEqual(float(sin_a ** 2 + cos_a ** 2), 1.0, places=5)

    def test_near_pi_and_near_negative_pi_land_close_in_encoded_space(self):
        # The discontinuity the sin/cos encoding was added to remove: under the old
        # linear angle/pi encoding, two physically-adjacent angles (just inside +pi
        # and just inside -pi, i.e. "directly behind the agent") landed maximally
        # far apart (~+1 vs ~-1). They should now land close together.
        offset = OWN_STATE_DIM + len(BIOME_TYPES)
        obs_a = [dict(type="Fruit", distance=30.0, angle=np.pi - 0.01)]
        obs_b = [dict(type="Fruit", distance=30.0, angle=-np.pi + 0.01)]
        vec_a = encode_observation(_make_status(obs_a), sim_time_frac=0.0)[offset:offset + FRUIT_SLOT_DIM]
        vec_b = encode_observation(_make_status(obs_b), sim_time_frac=0.0)[offset:offset + FRUIT_SLOT_DIM]
        self.assertLess(float(np.linalg.norm(vec_a[:3] - vec_b[:3])), 0.05)


class RealEnvironmentIntegrationTest(unittest.TestCase):
    """Sanity check the real call sites (SurvivalEnv.reset/step) still produce
    well-formed vectors end to end, not just under synthetic input."""

    def test_reset_and_step_produce_correctly_shaped_finite_observations(self):
        env = SurvivalEnv(starting_agents=2, starting_predators=0)
        obs, _ = env.reset(seed=1)
        for vec in obs.values():
            self.assertEqual(vec.shape, (OBS_DIM,))
            self.assertTrue(np.isfinite(vec).all())

        continuous = np.array([0.3, 0.0, 1.0, 0.0], dtype=np.float32)
        actions = {aid: (continuous, 0.0) for aid in obs}
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        for vec in next_obs.values():
            self.assertEqual(vec.shape, (OBS_DIM,))
            self.assertTrue(np.isfinite(vec).all())


if __name__ == "__main__":
    unittest.main()
