"""
Pure, fast unit tests for the reward formulas in gym_env.py - no SimulationCore
needed, since fruit_approach_reward/predator_avoid_reward/spawn_reward/
search_reward_and_ref/momentum_reward/predator_face_reward are all pure functions
of their inputs.
"""
import unittest

import numpy as np

from src.training import gym_env


class FruitApproachRewardTests(unittest.TestCase):
    def test_none_prev_gives_zero(self):
        self.assertEqual(gym_env.fruit_approach_reward(None, 50.0), 0.0)

    def test_none_curr_gives_zero(self):
        self.assertEqual(gym_env.fruit_approach_reward(50.0, None), 0.0)

    def test_both_none_gives_zero(self):
        self.assertEqual(gym_env.fruit_approach_reward(None, None), 0.0)

    def test_closing_distance_is_positive(self):
        self.assertGreater(gym_env.fruit_approach_reward(100.0, 80.0), 0.0)

    def test_opening_distance_is_negative(self):
        self.assertLess(gym_env.fruit_approach_reward(80.0, 100.0), 0.0)

    def test_magnitude_matches_formula(self):
        reward = gym_env.fruit_approach_reward(100.0, 80.0)
        expected = gym_env.FRUIT_APPROACH_COEF * (100.0 - 80.0) / gym_env.NORM_DIST
        self.assertAlmostEqual(reward, expected, places=6)

    def test_telescopes_along_a_non_monotonic_path(self):
        # Sum of per-tick rewards along any path should equal the reward of the
        # single start-to-end delta, regardless of how many steps it took or the
        # (non-monotonic) shape of the path in between - the defining property of
        # potential-based shaping, and what makes it immune to path-based farming.
        distances = [150.0, 130.0, 140.0, 90.0, 60.0]
        total = sum(
            gym_env.fruit_approach_reward(distances[i], distances[i + 1])
            for i in range(len(distances) - 1)
        )
        direct = gym_env.fruit_approach_reward(distances[0], distances[-1])
        self.assertAlmostEqual(total, direct, places=6)


class PredatorAvoidRewardTests(unittest.TestCase):
    def test_none_gives_zero(self):
        self.assertEqual(gym_env.predator_avoid_reward(None, 50.0), 0.0)
        self.assertEqual(gym_env.predator_avoid_reward(50.0, None), 0.0)

    def test_increasing_distance_is_positive(self):
        self.assertGreater(gym_env.predator_avoid_reward(50.0, 80.0), 0.0)

    def test_decreasing_distance_is_negative(self):
        self.assertLess(gym_env.predator_avoid_reward(80.0, 50.0), 0.0)

    def test_is_sign_flipped_mirror_of_fruit_approach(self):
        prev, curr = 80.0, 50.0
        self.assertAlmostEqual(
            gym_env.predator_avoid_reward(prev, curr),
            -gym_env.fruit_approach_reward(prev, curr),
            places=6,
        )


class SpawnRewardTests(unittest.TestCase):
    def test_at_or_below_gate_energy_gives_zero(self):
        self.assertEqual(gym_env.spawn_reward(100.0), 0.0)
        self.assertEqual(gym_env.spawn_reward(50.0), 0.0)
        self.assertEqual(gym_env.spawn_reward(0.0), 0.0)

    def test_barely_above_gate_gives_near_zero(self):
        # Barely-legal spawn (100.01 energy) should net a tiny fraction of the full
        # reward, not the full amount - this is exactly the reckless-breeding case
        # SPAWN_SAFETY_MARGIN scaling was added to stop being profitable.
        reward = gym_env.spawn_reward(100.01)
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, gym_env.SPAWN_REWARD * 0.01)

    def test_full_safety_margin_gives_full_reward(self):
        energy = 100.0 + gym_env.SPAWN_SAFETY_MARGIN
        self.assertAlmostEqual(gym_env.spawn_reward(energy), gym_env.SPAWN_REWARD, places=6)

    def test_beyond_safety_margin_still_caps_at_full_reward(self):
        self.assertAlmostEqual(gym_env.spawn_reward(1000.0), gym_env.SPAWN_REWARD, places=6)

    def test_scales_linearly_within_margin(self):
        half_margin_energy = 100.0 + gym_env.SPAWN_SAFETY_MARGIN / 2
        reward = gym_env.spawn_reward(half_margin_energy)
        self.assertAlmostEqual(reward, gym_env.SPAWN_REWARD / 2, places=6)


class SearchRewardAndRefTests(unittest.TestCase):
    def test_zero_gap_gives_zero_reward(self):
        reward, new_ref = gym_env.search_reward_and_ref((10.0, 10.0), (10.0, 10.0))
        self.assertEqual(reward, 0.0)
        self.assertEqual(new_ref, (10.0, 10.0))

    def test_positive_gap_gives_positive_reward(self):
        reward, _ = gym_env.search_reward_and_ref((110.0, 10.0), (10.0, 10.0))
        self.assertGreater(reward, 0.0)

    def test_new_ref_moves_toward_current_position_by_alpha_fraction(self):
        curr_pos = (110.0, 10.0)
        ref_pos = (10.0, 10.0)
        _, new_ref = gym_env.search_reward_and_ref(curr_pos, ref_pos)
        expected_x = ref_pos[0] + gym_env.SEARCH_EMA_ALPHA * (curr_pos[0] - ref_pos[0])
        self.assertAlmostEqual(new_ref[0], expected_x, places=6)
        self.assertAlmostEqual(new_ref[1], ref_pos[1], places=6)  # no y gap -> y unchanged

    def test_oscillation_stays_bounded_unlike_sustained_motion(self):
        # Alternating a fixed distance either side of a stationary reference is the
        # shape of exploit a since-removed SCAN_REWARD_COEF fell for. The EMA should
        # keep the oscillating total small/bounded, unlike genuine sustained motion
        # covering the same per-tick distance, which should keep growing.
        ref = (0.0, 0.0)
        oscillate_total = 0.0
        for i in range(200):
            pos = (10.0, 0.0) if i % 2 == 0 else (-10.0, 0.0)
            reward, ref = gym_env.search_reward_and_ref(pos, ref)
            oscillate_total += reward

        straight_ref = (0.0, 0.0)
        straight_pos = 0.0
        straight_total = 0.0
        for _ in range(200):
            straight_pos += 10.0
            reward, straight_ref = gym_env.search_reward_and_ref((straight_pos, 0.0), straight_ref)
            straight_total += reward

        self.assertLess(oscillate_total, straight_total)

    def test_sustained_motion_converges_to_predicted_steady_state_gap(self):
        # For a constant per-tick displacement d, the EMA gap has a closed-form fixed
        # point: g_{t+1} = (1-alpha)*(g_t + d), so g_ss = d*(1-alpha)/alpha. Verify
        # the actual iteration reaches it after enough ticks to settle.
        step = 2.0
        pos = 0.0
        ref = (0.0, 0.0)
        for _ in range(2000):
            pos += step
            _, ref = gym_env.search_reward_and_ref((pos, 0.0), ref)
        actual_gap = pos - ref[0]
        expected_gap = step * (1 - gym_env.SEARCH_EMA_ALPHA) / gym_env.SEARCH_EMA_ALPHA
        self.assertAlmostEqual(actual_gap, expected_gap, delta=expected_gap * 0.01)


class MomentumRewardTests(unittest.TestCase):
    def test_none_gives_zero(self):
        self.assertEqual(gym_env.momentum_reward(None, (1.0, 0.0)), 0.0)
        self.assertEqual(gym_env.momentum_reward((1.0, 0.0), None), 0.0)

    def test_zero_prev_displacement_gives_zero(self):
        self.assertEqual(gym_env.momentum_reward((0.0, 0.0), (1.0, 0.0)), 0.0)

    def test_zero_curr_displacement_gives_zero(self):
        self.assertEqual(gym_env.momentum_reward((1.0, 0.0), (0.0, 0.0)), 0.0)

    def test_same_direction_is_positive(self):
        self.assertGreater(gym_env.momentum_reward((5.0, 0.0), (5.0, 0.0)), 0.0)

    def test_opposite_direction_is_negative(self):
        self.assertLess(gym_env.momentum_reward((5.0, 0.0), (-5.0, 0.0)), 0.0)

    def test_perpendicular_direction_is_zero(self):
        self.assertAlmostEqual(gym_env.momentum_reward((5.0, 0.0), (0.0, 5.0)), 0.0, places=6)

    def test_scales_with_current_tick_magnitude(self):
        small = gym_env.momentum_reward((5.0, 0.0), (1.0, 0.0))
        large = gym_env.momentum_reward((5.0, 0.0), (10.0, 0.0))
        self.assertGreater(large, small)

    def test_invariant_to_shared_rotation(self):
        # Alignment is about the angle BETWEEN the two vectors, not their absolute
        # heading - rotating both by the same amount shouldn't change the reward.
        base = gym_env.momentum_reward((5.0, 0.0), (5.0, 2.0))
        theta = 1.3
        rot = lambda v: (v[0] * np.cos(theta) - v[1] * np.sin(theta), v[0] * np.sin(theta) + v[1] * np.cos(theta))
        rotated = gym_env.momentum_reward(rot((5.0, 0.0)), rot((5.0, 2.0)))
        self.assertAlmostEqual(base, rotated, places=6)


class PredatorFaceRewardTests(unittest.TestCase):
    def test_none_gives_zero(self):
        self.assertEqual(gym_env.predator_face_reward(None, 0.5, 200.0), 0.0)
        self.assertEqual(gym_env.predator_face_reward(0.5, None, 200.0), 0.0)
        self.assertEqual(gym_env.predator_face_reward(0.5, 0.3, None), 0.0)

    def test_within_charge_range_gives_zero_even_if_facing_improved(self):
        reward = gym_env.predator_face_reward(1.0, 0.1, gym_env.PREDATOR_CHARGE_RANGE - 1.0)
        self.assertEqual(reward, 0.0)

    def test_at_charge_range_boundary_gives_zero(self):
        reward = gym_env.predator_face_reward(1.0, 0.1, gym_env.PREDATOR_CHARGE_RANGE)
        self.assertEqual(reward, 0.0)

    def test_turning_toward_predator_beyond_range_is_positive(self):
        reward = gym_env.predator_face_reward(1.0, 0.2, gym_env.PREDATOR_CHARGE_RANGE + 1.0)
        self.assertGreater(reward, 0.0)

    def test_turning_away_from_predator_beyond_range_is_negative(self):
        reward = gym_env.predator_face_reward(0.2, 1.0, gym_env.PREDATOR_CHARGE_RANGE + 1.0)
        self.assertLess(reward, 0.0)

    def test_magnitude_matches_formula(self):
        prev_angle, curr_angle, dist = 1.0, 0.2, 150.0
        reward = gym_env.predator_face_reward(prev_angle, curr_angle, dist)
        expected = gym_env.PREDATOR_FACE_COEF * (abs(prev_angle) - abs(curr_angle)) / np.pi
        self.assertAlmostEqual(reward, expected, places=6)

    def test_telescopes_along_a_non_monotonic_path_beyond_range(self):
        angles = [1.0, 0.6, 0.9, 0.3, -0.2]
        dist = gym_env.PREDATOR_CHARGE_RANGE + 5.0
        total = sum(
            gym_env.predator_face_reward(angles[i], angles[i + 1], dist)
            for i in range(len(angles) - 1)
        )
        direct = gym_env.predator_face_reward(angles[0], angles[-1], dist)
        self.assertAlmostEqual(total, direct, places=6)

    def test_oscillating_facing_direction_nets_to_zero(self):
        # A removed FRUIT_DISCOVERY_BONUS-style exploit farmed reward by sweeping
        # facing back and forth. Potential-based shaping should make any full
        # back-and-forth cycle in the angle net to exactly zero.
        dist = gym_env.PREDATOR_CHARGE_RANGE + 5.0
        angle_sequence = [1.0, 0.2, 1.2, 0.1, 1.0]  # ends where it started
        total = sum(
            gym_env.predator_face_reward(angle_sequence[i], angle_sequence[i + 1], dist)
            for i in range(len(angle_sequence) - 1)
        )
        self.assertAlmostEqual(total, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
