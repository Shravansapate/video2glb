from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import json
import math
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
    timestamps_seconds: list[float] = field(default_factory=list)
    variable_frame_rate: bool = False
    timestamp_source: str = "nominal_fps"
    reported_frame_count: int | None = None
    fps_rational: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedVideo:
    source_info: VideoInfo
    working_info: VideoInfo
    working_path: Path
    normalized: bool
    timing_mapping: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_info.to_json_dict(),
            "working": self.working_info.to_json_dict(),
            "working_path": str(self.working_path),
            "normalized": self.normalized,
            "normalization": "ffmpeg_fps_round_near_lossless_ffv1" if self.normalized else "none",
            "timing_mapping": self.timing_mapping,
        }


def inspect_video(video_path: str | Path) -> VideoInfo:
    """Decode every frame; container frame counts alone are never evidence."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    ffprobe = shutil.which("ffprobe")
    probe = _inspect_with_ffprobe(path, ffprobe) if ffprobe else None
    return _inspect_with_opencv(path, probe)


def _inspect_with_ffprobe(path: Path, ffprobe: str) -> dict[str, Any] | None:
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:"
        "stream_tags=rotate:stream_side_data=rotation:"
        "frame=best_effort_timestamp_time,pts_time,duration_time,pkt_duration_time",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        data = json.loads(result.stdout)
    except OSError:
        return None
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        detail = (getattr(exc, "stderr", "") or str(exc))[-1000:]
        raise RuntimeError(f"ffprobe could not decode and inspect the source video: {detail}") from exc
    if not isinstance(data, dict) or not data.get("streams"):
        raise RuntimeError(f"ffprobe did not return a video stream: {path}")
    if not isinstance(data["streams"], list) or not isinstance(data["streams"][0], dict):
        raise RuntimeError(f"ffprobe returned malformed stream metadata: {path}")
    if result.stderr.strip():
        raise RuntimeError(f"Video decode errors reported by ffprobe for {path}: {result.stderr.strip()[:500]}")
    return data


def _inspect_with_opencv(path: Path, probe: dict[str, Any] | None = None) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        reported_count = _safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
        codec = _fourcc_to_string(_safe_int(cap.get(cv2.CAP_PROP_FOURCC)) or 0)
        decoded_pts: list[float] = []
        width = height = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim < 2:
                raise RuntimeError(f"Invalid decoded frame {len(decoded_pts)} in {path}")
            current_height, current_width = frame.shape[:2]
            if decoded_pts and (current_width, current_height) != (width, height):
                raise RuntimeError(f"Video resolution changes during decoding: {path}")
            width, height = current_width, current_height
            decoded_pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
        count = len(decoded_pts)
        if count == 0:
            raise RuntimeError(f"Video contains no decodable frames: {path}")

        stream = (probe.get("streams") or [{}])[0] if probe else {}
        probe_frames = probe.get("frames") if probe else None
        if probe_frames is not None:
            if not isinstance(probe_frames, list) or len(probe_frames) != count:
                raise RuntimeError(f"Video decode frame mismatch: OpenCV decoded {count}, ffprobe decoded {len(probe_frames or [])}: {path}")
            if not all(isinstance(frame, dict) for frame in probe_frames):
                raise RuntimeError(f"ffprobe returned malformed frame metadata: {path}")
            timestamps = [_safe_float(frame.get("best_effort_timestamp_time", frame.get("pts_time"))) for frame in probe_frames]
            timestamp_source = "ffprobe_decoded_pts"
        else:
            timestamps = decoded_pts
            timestamp_source = "opencv_decoded_pts"
        claimed_count = _safe_int(stream.get("nb_frames"))
        if claimed_count is not None and claimed_count > 0:
            discrepancy = abs(claimed_count - count)
            if discrepancy > 1 or (discrepancy == 1 and claimed_count < 10):
                raise RuntimeError(f"Incomplete video decode: container declares {claimed_count} frames, decoded {count}: {path}")
        if not _valid_timestamps(timestamps):
            raise RuntimeError(f"Video lacks finite, strictly increasing decoded timestamps; install a working ffprobe or repair the source: {path}")
        origin = float(timestamps[0])
        relative = [float(value) - origin for value in timestamps]
        average_fps = _parse_fps(stream.get("avg_frame_rate"))
        nominal_fps = _parse_fps(stream.get("r_frame_rate"))
        fps = average_fps or cap_fps or nominal_fps
        if fps is None or not math.isfinite(fps) or fps <= 0 or fps > 1000:
            raise RuntimeError(f"Could not determine a supported finite FPS (0 < FPS <= 1000): {path}")
        intervals = [right - left for left, right in zip(relative, relative[1:])]
        variable = _is_variable_timing(relative, fps)
        if variable:
            ordered = sorted(intervals)
            median = ordered[len(ordered) // 2] if ordered else 1.0 / fps
            fps = 1.0 / median
            if nominal_fps and abs(nominal_fps - fps) / fps < 0.005:
                fps = nominal_fps
            if not math.isfinite(fps) or fps <= 0 or fps > 1000:
                raise RuntimeError(f"Decoded video cadence exceeds supported tracking FPS: {path}")
        rational = str(Fraction(fps).limit_denominator(100000))
        rotation = _safe_int((stream.get("tags") or {}).get("rotate"))
        for side_data in stream.get("side_data_list") or []:
            rotation = _safe_int(side_data.get("rotation")) if "rotation" in side_data else rotation
        final_duration = 1.0 / fps
        if probe_frames:
            last = probe_frames[-1]
            candidate = _safe_float(last.get("duration_time", last.get("pkt_duration_time")))
            if candidate is not None and candidate > 0:
                final_duration = candidate
        return VideoInfo(
            path=str(path), codec=str(stream.get("codec_name") or codec), width=width, height=height,
            fps=fps, duration_seconds=relative[-1] + final_duration, frame_count=count,
            rotation_degrees=rotation, timestamps_ms=_milliseconds(relative),
            inspector="ffprobe+opencv_decode" if probe else "opencv_decode",
            timestamps_seconds=relative, variable_frame_rate=variable,
            timestamp_source=timestamp_source, reported_frame_count=claimed_count or reported_count,
            fps_rational=rational,
        )
    finally:
        cap.release()


def prepare_video(video_path: str | Path, working_directory: str | Path) -> PreparedVideo:
    """Keep the source intact and normalize VFR to a verified lossless CFR copy."""
    source = inspect_video(video_path)
    working_path = Path(video_path)
    working = source
    if source.variable_frame_rate:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Variable-frame-rate input requires a working FFmpeg executable on PATH.")
        directory = Path(working_directory)
        directory.mkdir(parents=True, exist_ok=True)
        working_path = directory / "working_cfr.mkv"
        if working_path.resolve() == Path(video_path).resolve():
            raise RuntimeError("Working video path must not overwrite the original input.")
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video_path),
            "-map", "0:v:0", "-an", "-vf",
            f"setpts=PTS-STARTPTS,fps=fps={source.fps_rational}:start_time=0:round=near",
            "-fps_mode", "cfr", "-c:v", "ffv1", "-threads", "1", str(working_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = (getattr(exc, "stderr", "") or str(exc))[-1000:]
            raise RuntimeError(f"FFmpeg VFR normalization failed: {detail}") from exc
        working = inspect_video(working_path)
        if working.variable_frame_rate or abs(working.fps - source.fps) / source.fps > 0.001:
            raise RuntimeError("FFmpeg working video did not preserve the requested constant frame rate.")
        if abs(working.duration_seconds - source.duration_seconds) > 1.5 / source.fps:
            raise RuntimeError("FFmpeg working video duration differs from source by more than 1.5 frames.")
    source_ticks = [math.floor(value * working.fps + 0.5) for value in source.timestamps_seconds]
    mapping = []
    for index, timestamp in enumerate(working.timestamps_seconds):
        source_index = min(len(source_ticks) - 1, max(0, bisect_right(source_ticks, index) - 1)) if source.variable_frame_rate else index
        mapping.append({"working_frame": index, "working_timestamp_seconds": timestamp,
                        "source_frame": source_index, "source_timestamp_seconds": source.timestamps_seconds[source_index]})
    return PreparedVideo(source, working, working_path, source.variable_frame_rate, mapping)


def _valid_timestamps(values: list[Any]) -> bool:
    return bool(values) and all(value is not None and math.isfinite(value) for value in values) and all(right > left for left, right in zip(values, values[1:]))


def _is_variable_timing(timestamps: list[float], fps: float) -> bool:
    if len(timestamps) < 2:
        return False
    period = 1.0 / fps
    # Allow timestamp rounding in millisecond containers, not accumulated drift.
    tolerance = max(0.0011, period * 0.01)
    return any(abs(value - index * period) > tolerance for index, value in enumerate(timestamps))


def _parse_fps(value: Any) -> float | None:
    try:
        result = float(Fraction(str(value)))
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _milliseconds(timestamps: list[float]) -> list[int]:
    values = [round(value * 1000) for value in timestamps]
    if any(right <= left for left, right in zip(values, values[1:])):
        raise RuntimeError("Video frame timestamps are too close for millisecond tracking timestamps.")
    return values


def _timestamps_ms(frame_count: int, fps: float) -> list[int]:
    return _milliseconds([index / fps for index in range(frame_count)])


def _fourcc_to_string(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ") or "unknown"


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None
