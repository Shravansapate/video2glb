from __future__ import annotations

import math
import numpy as np


class OneEuroFilter:
    def __init__(self, fps: float, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.frequency = fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.previous: np.ndarray | None = None
        self.previous_derivative: np.ndarray | None = None

    def apply(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.previous is None:
            self.previous = value
            self.previous_derivative = np.zeros_like(value)
            return value
        derivative = (value - self.previous) * self.frequency
        derivative_hat = exponential_smooth(self.alpha(self.d_cutoff), derivative, self.previous_derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(derivative_hat)
        value_hat = exponential_smooth(self.alpha(cutoff), value, self.previous)
        self.previous = value_hat
        self.previous_derivative = derivative_hat
        return value_hat

    def alpha(self, cutoff) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.frequency
        return 1.0 / (1.0 + tau / te)


def smooth_landmarks(values: np.ndarray, fps: float, min_cutoff: float, beta: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    frame_count = output.shape[0]
    flat = output.reshape(frame_count, -1)
    filters = [OneEuroFilter(fps=fps, min_cutoff=min_cutoff, beta=beta) for _ in range(flat.shape[1])]
    for frame_index in range(frame_count):
        for column, one_euro in enumerate(filters):
            if np.isfinite(flat[frame_index, column]):
                flat[frame_index, column] = one_euro.apply(np.array(flat[frame_index, column]))
    return output


def exponential_smooth(alpha, value, previous):
    return alpha * value + (1.0 - alpha) * previous
