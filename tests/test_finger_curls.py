import math
import unittest

import numpy as np

from src.motion.skeleton_solver import (
    _build_finger_directions,
    _compute_finger_curls,
    _hold_last_finger_curls,
    _hold_finger_directions_for_release,
    _limit_finger_direction_steps,
    _limit_finger_curl_steps,
    _ramp_finger_direction_activation,
)


class FingerCurlTests(unittest.TestCase):
    def test_index_curl_is_zero_when_straight(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)
        hand[0, 0] = [0.0, 0.0, 0.0]
        hand[0, 5] = [1.0, 0.0, 0.0]
        hand[0, 6] = [2.0, 0.0, 0.0]
        hand[0, 7] = [3.0, 0.0, 0.0]
        hand[0, 8] = [4.0, 0.0, 0.0]

        curls = _compute_finger_curls(hand)

        self.assertAlmostEqual(float(curls[0, 1, 0]), 0.0, places=7)
        self.assertAlmostEqual(float(curls[0, 1, 1]), 0.0, places=7)
        self.assertAlmostEqual(float(curls[0, 1, 2]), 0.0, places=7)

    def test_index_curl_reports_ninety_degree_bend(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)
        hand[0, 0] = [0.0, 0.0, 0.0]
        hand[0, 5] = [1.0, 0.0, 0.0]
        hand[0, 6] = [1.0, 1.0, 0.0]
        hand[0, 7] = [1.0, 2.0, 0.0]
        hand[0, 8] = [1.0, 3.0, 0.0]

        curls = _compute_finger_curls(hand)

        self.assertAlmostEqual(float(curls[0, 1, 0]), math.pi / 2.0, places=7)

    def test_finger_curl_step_limiter_preserves_gaps_and_limits_continuous_motion(self):
        curls = np.array([[[0.0]], [[math.pi]], [[math.pi]], [[np.nan]], [[math.pi]]], dtype=np.float64)

        limited = _limit_finger_curl_steps(curls, max_delta=math.pi / 4.0)

        self.assertAlmostEqual(float(limited[1, 0, 0]), math.pi / 4.0, places=7)
        self.assertAlmostEqual(float(limited[2, 0, 0]), math.pi / 2.0, places=7)
        self.assertTrue(math.isnan(float(limited[3, 0, 0])))
        self.assertAlmostEqual(float(limited[4, 0, 0]), math.pi / 4.0, places=7)

    def test_missing_finger_observation_holds_the_last_known_handshape(self):
        curls = np.array([[[0.2]], [[np.nan]], [[np.nan]], [[0.5]]], dtype=np.float64)

        held = _hold_last_finger_curls(curls)

        self.assertAlmostEqual(float(held[1, 0, 0]), 0.2, places=7)
        self.assertAlmostEqual(float(held[2, 0, 0]), 0.2, places=7)
        self.assertAlmostEqual(float(held[3, 0, 0]), 0.5, places=7)

    def test_finger_directions_are_expressed_in_the_local_palm_basis(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)
        hand[0, 0] = [0.0, 0.0, 0.0]
        hand[0, 5] = [1.0, 0.0, 0.0]
        hand[0, 6] = [1.0, 1.0, 0.0]
        hand[0, 7] = [1.0, 2.0, 0.0]
        hand[0, 8] = [1.0, 3.0, 0.0]
        hand[0, 9] = [0.0, 1.0, 0.0]
        hand[0, 17] = [-1.0, 0.0, 0.0]

        directions, valid = _build_finger_directions(hand)

        self.assertTrue(bool(valid[0, 1, 0]))
        np.testing.assert_allclose(directions[0, 1, 0], [0.0, 1.0, 0.0], atol=1e-7)

    def test_finger_directions_remain_invalid_without_a_palm_basis(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)

        _, valid = _build_finger_directions(hand)

        self.assertFalse(bool(valid.any()))

    def test_finger_direction_step_limiter_caps_a_large_turn(self):
        directions = np.full((2, 1, 1, 1, 3), np.nan, dtype=np.float64)
        directions[0, 0, 0, 0] = [1.0, 0.0, 0.0]
        directions[1, 0, 0, 0] = [0.0, 1.0, 0.0]
        valid = np.ones((2, 1, 1, 1), dtype=bool)

        limited = _limit_finger_direction_steps(directions, valid, max_delta=math.pi / 6.0)

        angle = math.degrees(math.acos(float(np.clip(np.dot(limited[0, 0, 0, 0], limited[1, 0, 0, 0]), -1.0, 1.0))))
        self.assertAlmostEqual(angle, 30.0, places=6)

    def test_reacquired_finger_direction_fades_in(self):
        valid = np.array([[[[False]]], [[[True]]], [[[True]]], [[[True]]], [[[True]]]], dtype=bool)

        influence = _ramp_finger_direction_activation(valid, valid, ramp_frames=4)

        np.testing.assert_allclose(influence[:, 0, 0, 0], [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_lost_finger_direction_holds_then_fades_out(self):
        directions = np.full((6, 1, 1, 1, 3), np.nan, dtype=np.float64)
        directions[:2, 0, 0, 0] = [1.0, 0.0, 0.0]
        source_valid = np.array([[[[True]]], [[[True]]], [[[False]]], [[[False]]], [[[False]]], [[[False]]]], dtype=bool)

        held, constraint_valid = _hold_finger_directions_for_release(directions, source_valid, release_frames=3)
        influence = _ramp_finger_direction_activation(source_valid, constraint_valid, ramp_frames=3)

        self.assertTrue(bool(constraint_valid[2, 0, 0, 0]))
        self.assertTrue(bool(constraint_valid[4, 0, 0, 0]))
        self.assertFalse(bool(constraint_valid[5, 0, 0, 0]))
        np.testing.assert_allclose(held[2, 0, 0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(influence[:, 0, 0, 0], [1 / 3, 2 / 3, 5 / 12, 1 / 6, 0.0, 0.0])
