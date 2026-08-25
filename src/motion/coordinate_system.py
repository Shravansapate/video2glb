from __future__ import annotations

import numpy as np


def mediapipe_world_to_canonical(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    converted = points[..., :3].copy()
    x = converted[..., 0].copy()
    y = converted[..., 1].copy()
    z = converted[..., 2].copy()
    converted[..., 0] = x
    converted[..., 1] = -y
    converted[..., 2] = -z
    return converted


def mediapipe_image_to_canonical(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    converted = np.zeros(points.shape[:-1] + (3,), dtype=np.float64)
    converted[..., 0] = points[..., 0] - 0.5
    converted[..., 1] = 0.5 - points[..., 1]
    converted[..., 2] = -points[..., 2]
    return converted


def canonical_to_blender_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector[..., :3].copy()


def gltf_quaternion_order() -> str:
    return "XYZW"


def internal_quaternion_order() -> str:
    return "WXYZ"
