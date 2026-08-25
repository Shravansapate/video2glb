from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.tracking.pose_schema import PoseSequence


LEFT_WRIST = 15
RIGHT_WRIST = 16
HAND_WRIST = 0


@dataclass(frozen=True)
class TrackingQcResult:
    technical_qc: str
    isl_verified: bool
    metrics: dict[str, Any]
    review_reasons: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "technical_qc": self.technical_qc,
            "isl_verified": self.isl_verified,
            "metrics": self.metrics,
            "review_reasons": self.review_reasons,
        }


def evaluate_tracking(sequence: PoseSequence, thresholds: dict[str, Any]) -> TrackingQcResult:
    frames = sequence.frames
    frame_count = len(frames)
    if frame_count == 0:
        return TrackingQcResult(
            technical_qc="FAIL",
            isl_verified=False,
            metrics={"frame_count": 0},
            review_reasons=["No frames were processed."],
        )

    pose_present = np.array([frame.tracking_quality.get("pose_present", False) for frame in frames], dtype=bool)
    left_present = np.array([frame.tracking_quality.get("left_hand_present", False) for frame in frames], dtype=bool)
    right_present = np.array([frame.tracking_quality.get("right_hand_present", False) for frame in frames], dtype=bool)
    any_hand_present = left_present | right_present
    swap_events = _count_possible_swap_events(sequence)

    longest_missing_hand_frames = _longest_false_run(any_hand_present)
    longest_missing_hand_seconds = longest_missing_hand_frames / sequence.fps if sequence.fps else 0.0

    metrics = {
        "frame_count": frame_count,
        "fps": sequence.fps,
        "body_tracking_availability": float(pose_present.mean()),
        "left_hand_availability": float(left_present.mean()),
        "right_hand_availability": float(right_present.mean()),
        "longest_missing_hand_sequence_frames": int(longest_missing_hand_frames),
        "longest_missing_hand_sequence_seconds": float(longest_missing_hand_seconds),
        "left_right_swap_events": int(swap_events),
    }

    tracking_thresholds = thresholds.get("tracking", thresholds)
    review_reasons: list[str] = []
    status = "PASS"

    if metrics["body_tracking_availability"] < float(tracking_thresholds.get("body_availability_fail_below", 0.5)):
        status = "FAIL"
        review_reasons.append("Body tracking availability is below the fail threshold.")
    elif metrics["body_tracking_availability"] < float(tracking_thresholds.get("body_availability_review_below", 0.8)):
        status = "REVIEW"
        review_reasons.append("Body tracking availability is below the review threshold.")

    hand_review_threshold = float(tracking_thresholds.get("hand_availability_review_below", 0.2))
    if metrics["left_hand_availability"] < hand_review_threshold:
        status = _max_status(status, "REVIEW")
        review_reasons.append("Left hand tracking availability is low.")
    if metrics["right_hand_availability"] < hand_review_threshold:
        status = _max_status(status, "REVIEW")
        review_reasons.append("Right hand tracking availability is low.")

    if longest_missing_hand_seconds > float(tracking_thresholds.get("longest_missing_hand_review_seconds", 1.5)):
        status = _max_status(status, "REVIEW")
        review_reasons.append("A long missing-hand sequence was detected.")

    if swap_events >= int(tracking_thresholds.get("swap_events_review_at_or_above", 1)):
        status = _max_status(status, "REVIEW")
        review_reasons.append("Possible left/right hand swap events were detected.")

    return TrackingQcResult(
        technical_qc=status,
        isl_verified=False,
        metrics=metrics,
        review_reasons=review_reasons,
    )


def _count_possible_swap_events(sequence: PoseSequence) -> int:
    events = 0
    for frame in sequence.frames:
        pose = frame.pose_image
        if frame.left_hand_image is None or frame.right_hand_image is None:
            continue
        if not (
            np.isfinite(pose[[LEFT_WRIST, RIGHT_WRIST], :2]).all()
            and np.isfinite(frame.left_hand_image[HAND_WRIST, :2]).all()
            and np.isfinite(frame.right_hand_image[HAND_WRIST, :2]).all()
        ):
            continue

        left_pose = pose[LEFT_WRIST, :2]
        right_pose = pose[RIGHT_WRIST, :2]
        left_hand = frame.left_hand_image[HAND_WRIST, :2]
        right_hand = frame.right_hand_image[HAND_WRIST, :2]

        normal_distance = np.linalg.norm(left_hand - left_pose) + np.linalg.norm(right_hand - right_pose)
        swapped_distance = np.linalg.norm(left_hand - right_pose) + np.linalg.norm(right_hand - left_pose)
        if swapped_distance + 0.02 < normal_distance:
            events += 1
    return events


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def _max_status(current: str, candidate: str) -> str:
    order = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
    return candidate if order[candidate] > order[current] else current
