from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield frame_index, frame_bgr
            frame_index += 1
    finally:
        cap.release()
