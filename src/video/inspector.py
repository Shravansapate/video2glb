from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    rotation_degrees: int | None
    timestamps_ms: list[int]
    inspector: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probed = _inspect_with_ffprobe(path, ffprobe)
        if probed is not None:
            return probed

    return _inspect_with_opencv(path)


def _inspect_with_ffprobe(path: Path, ffprobe: str) -> VideoInfo | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:stream_tags=rotate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None

    streams = data.get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))
    frame_count = _safe_int(stream.get("nb_frames"))
    duration = _safe_float(stream.get("duration"))

    if fps is None or fps <= 0:
        return None
    if frame_count is None:
        frame_count = int(round((duration or 0) * fps))
    if duration is None and frame_count:
        duration = frame_count / fps

    return VideoInfo(
        path=str(path),
        codec=str(stream.get("codec_name") or "unknown"),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=float(fps),
        duration_seconds=float(duration or 0.0),
        frame_count=int(frame_count or 0),
        rotation_degrees=_safe_int((stream.get("tags") or {}).get("rotate")),
        timestamps_ms=_timestamps_ms(int(frame_count or 0), float(fps)),
        inspector="ffprobe",
    )


def _inspect_with_opencv(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        codec = _fourcc_to_string(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))

        if fps <= 0:
            raise RuntimeError(f"Could not determine FPS for video: {path}")

        return VideoInfo(
            path=str(path),
            codec=codec,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=frame_count / fps if frame_count else 0.0,
            frame_count=frame_count,
            rotation_degrees=None,
            timestamps_ms=_timestamps_ms(frame_count, fps),
            inspector="opencv",
        )
    finally:
        cap.release()


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    return float(value)


def _timestamps_ms(frame_count: int, fps: float) -> list[int]:
    timestamps: list[int] = []
    previous = -1
    for frame_index in range(frame_count):
        timestamp = int(round((frame_index / fps) * 1000.0))
        if timestamp <= previous:
            timestamp = previous + 1
        timestamps.append(timestamp)
        previous = timestamp
    return timestamps


def _fourcc_to_string(value: int) -> str:
    chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
    codec = "".join(chars).strip()
    return codec or "unknown"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
