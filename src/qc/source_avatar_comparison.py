from __future__ import annotations

from pathlib import Path

import cv2


def create_source_avatar_comparison(
    source_video: str | Path,
    avatar_preview: str | Path,
    output_video: str | Path,
) -> dict:
    source_path = Path(source_video)
    avatar_path = Path(avatar_preview)
    output_path = Path(output_video)
    source = cv2.VideoCapture(str(source_path))
    avatar = cv2.VideoCapture(str(avatar_path))
    try:
        if not source.isOpened() or not avatar.isOpened():
            raise RuntimeError("Could not open source or avatar preview for comparison.")
        source_fps = float(source.get(cv2.CAP_PROP_FPS) or 0.0)
        avatar_fps = float(avatar.get(cv2.CAP_PROP_FPS) or 0.0)
        source_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        avatar_count = int(avatar.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0 or avatar_fps <= 0:
            raise RuntimeError("Comparison videos do not provide a valid FPS.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        target_height = min(540, max(source_height, 1))
        source_width_scaled = max(1, round(source_width * target_height / max(source_height, 1)))
        avatar_width_scaled = target_height
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (source_width_scaled + avatar_width_scaled, target_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create comparison video: {output_path}")
        written = 0
        try:
            while True:
                source_ok, source_frame = source.read()
                avatar_ok, avatar_frame = avatar.read()
                if not source_ok or not avatar_ok:
                    break
                left = cv2.resize(source_frame, (source_width_scaled, target_height), interpolation=cv2.INTER_AREA)
                right = cv2.resize(avatar_frame, (avatar_width_scaled, target_height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.hconcat([left, right]))
                written += 1
        finally:
            writer.release()

        source_duration = (source_count - 1) / source_fps if source_count else 0.0
        avatar_duration = (avatar_count - 1) / avatar_fps if avatar_count else 0.0
        status = "PASS" if abs(source_fps - avatar_fps) < 0.01 and abs(source_count - avatar_count) <= 1 else "REVIEW"
        return {
            "status": status,
            "source_frame_count": source_count,
            "avatar_frame_count": avatar_count,
            "comparison_frame_count": written,
            "source_fps": source_fps,
            "avatar_fps": avatar_fps,
            "source_duration_seconds": source_duration,
            "avatar_duration_seconds": avatar_duration,
            "duration_difference_seconds": abs(source_duration - avatar_duration),
            "technical_motion_match": "See Passenger.glb_validation.json for endpoint and finger-direction metrics.",
            "isl_verified": False,
            "isl_verification_status": "PENDING_SIGNER_REVIEW",
        }
    finally:
        source.release()
        avatar.release()
