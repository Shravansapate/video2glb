import math
import unittest

import numpy as np

from src.motion.skeleton_solver import (
    _build_finger_directions,
    _build_palm_bases,
    _condition_palm_bases,
    _compute_finger_curls,
    _hold_palm_bases_for_release,
    _hold_last_finger_curls,
    _hold_finger_directions_for_release,
    _limit_finger_direction_steps,
    _limit_finger_curl_steps,
    _quaternion_geodesic_angle,
    _ramp_finger_direction_activation,
    _ramp_palm_basis_activation,
)
from src.motion.quaternion_utils import quaternion_from_matrix


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

    def test_palm_basis_is_right_handed_and_orthonormal(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)
        hand[0, 0] = [0.0, 0.0, 0.0]
        hand[0, 5] = [1.0, 1.0, 0.0]
        hand[0, 9] = [0.0, 1.0, 0.0]
        hand[0, 17] = [-1.0, 1.0, 0.0]

        bases, valid = _build_palm_bases(hand, hand)

        self.assertTrue(bool(valid.all()))
        for basis in bases[0]:
            np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=7)

    def test_palm_basis_preserves_directed_anatomical_across_axis(self):
        hands = np.stack([self._make_flat_hand(0.0), self._make_flat_hand(math.pi)], axis=0)

        bases, observed = _build_palm_bases(hands, hands)

        self.assertTrue(bool(observed.all()))
        np.testing.assert_allclose(bases[0, 0, :, 0], [1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(bases[1, 0, :, 0], [-1.0, 0.0, 0.0], atol=1e-7)

    def test_palm_basis_rejects_nearly_collinear_palm_axes(self):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float64)
        hand[0, 0] = [0.0, 0.0, 0.0]
        hand[0, 5] = [1.0, 0.01, 0.0]
        hand[0, 9] = [1.0, 0.0, 0.0]
        hand[0, 17] = [-1.0, -0.01, 0.0]

        _, observed = _build_palm_bases(hand, hand)

        self.assertFalse(bool(observed.any()))

    def test_palm_conditioning_slerps_only_short_internal_gaps(self):
        bases = np.tile(np.eye(3), (5, 2, 1, 1)).astype(np.float64)
        bases[2, 0] = self._z_rotation(40.0)
        observed = np.zeros((5, 2), dtype=bool)
        observed[[0, 2], 0] = True

        conditioned, usable = _condition_palm_bases(
            bases,
            observed,
            max_gap=1,
            smoothing_radius=0,
            max_step=math.radians(35.0),
        )

        self.assertTrue(bool(usable[:3, 0].all()))
        self.assertFalse(bool(usable[3:, 0].any()))
        middle = quaternion_from_matrix(conditioned[1, 0])
        self.assertAlmostEqual(math.degrees(_quaternion_geodesic_angle([1, 0, 0, 0], middle)), 20.0, places=6)

    def test_palm_conditioning_does_not_fill_long_or_boundary_gaps(self):
        bases = np.tile(np.eye(3), (6, 2, 1, 1)).astype(np.float64)
        observed = np.zeros((6, 2), dtype=bool)
        observed[[1, 4], 0] = True

        _, usable = _condition_palm_bases(
            bases,
            observed,
            max_gap=1,
            smoothing_radius=0,
            max_step=math.radians(35.0),
        )

        np.testing.assert_array_equal(usable[:, 0], [False, True, False, False, True, False])

    def test_palm_conditioning_robustly_rejects_a_single_orientation_spike(self):
        bases = np.tile(np.eye(3), (5, 2, 1, 1)).astype(np.float64)
        bases[2, 0] = self._z_rotation(170.0)
        observed = np.zeros((5, 2), dtype=bool)
        observed[:, 0] = True

        conditioned, _ = _condition_palm_bases(
            bases,
            observed,
            max_gap=0,
            smoothing_radius=2,
            max_step=math.radians(35.0),
        )

        center = quaternion_from_matrix(conditioned[2, 0])
        self.assertLess(math.degrees(_quaternion_geodesic_angle([1, 0, 0, 0], center)), 1.0)

    def test_palm_conditioning_caps_each_consecutive_geodesic_step(self):
        bases = np.tile(np.eye(3), (3, 2, 1, 1)).astype(np.float64)
        bases[1, 0] = self._z_rotation(120.0)
        bases[2, 0] = self._z_rotation(120.0)
        observed = np.zeros((3, 2), dtype=bool)
        observed[:, 0] = True

        conditioned, usable = _condition_palm_bases(
            bases,
            observed,
            max_gap=0,
            smoothing_radius=0,
            max_step=math.radians(35.0),
        )
        quaternions = [quaternion_from_matrix(conditioned[index, 0]) for index in range(3)]
        steps = [math.degrees(_quaternion_geodesic_angle(quaternions[index - 1], quaternions[index])) for index in range(1, 3)]

        self.assertTrue(bool(usable[:, 0].all()))
        self.assertLessEqual(max(steps), 35.0 + 1e-7)

    def test_palm_release_uses_finite_last_basis_while_influence_fades(self):
        bases = np.tile(np.eye(3), (5, 2, 1, 1)).astype(np.float64)
        bases[0, 0] = self._z_rotation(25.0)
        usable = np.zeros((5, 2), dtype=bool)
        usable[0, 0] = True

        held, constraint_valid = _hold_palm_bases_for_release(bases, usable, release_frames=2)
        influence = _ramp_palm_basis_activation(usable, ramp_frames=2, constraint_valid=constraint_valid)

        np.testing.assert_allclose(held[1, 0], bases[0, 0], atol=1e-7)
        np.testing.assert_allclose(held[2, 0], bases[0, 0], atol=1e-7)
        self.assertTrue(bool(np.isfinite(held[:3, 0]).all()))
        np.testing.assert_array_equal(constraint_valid[:, 0], [True, True, True, False, False])
        np.testing.assert_allclose(influence[:, 0], [0.5, 1 / 6, 0.0, 0.0, 0.0])

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

    @staticmethod
    def _z_rotation(degrees: float) -> np.ndarray:
        angle = math.radians(degrees)
        return np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _make_flat_hand(cls, angle: float) -> np.ndarray:
        rotation = cls._z_rotation(math.degrees(angle))
        hand = np.full((21, 3), np.nan, dtype=np.float64)
        hand[0] = rotation @ np.array([0.0, 0.0, 0.0])
        hand[5] = rotation @ np.array([1.0, 1.0, 0.0])
        hand[9] = rotation @ np.array([0.0, 1.0, 0.0])
        hand[17] = rotation @ np.array([-1.0, 1.0, 0.0])
        return hand
