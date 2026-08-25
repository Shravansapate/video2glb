from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.avatar.bone_mapping import load_avatar_profile, load_bone_map
from src.motion.coordinate_system import mediapipe_image_to_canonical, mediapipe_world_to_canonical
from src.motion.interpolation import interpolate_short_gaps
from src.motion.quaternion_utils import (
    enforce_quaternion_continuity,
    normalize_quaternion,
    normalize_vector,
    limit_quaternion_angle,
    quaternion_from_axis_angle,
    quaternion_from_matrix,
    quaternion_slerp,
    quaternion_from_vectors,
    quaternion_to_matrix,
    validate_quaternions,
)
from src.motion.smoothing import smooth_landmarks


LEFT = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
}
RIGHT = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
}
HAND_CHAINS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Little": [17, 18, 19, 20],
}


@dataclass(frozen=True)
class MotionBuildResult:
    motion_path: Path
    metadata: dict


def solve_motion_from_pose(
    pose_path: str | Path,
    profile_path: str | Path,
    bone_map_path: str | Path,
    output_path: str | Path,
    gloss: str,
    body_only: bool = False,
    include_palms: bool = True,
    include_fingers: bool = True,
    smoothing: bool = True,
) -> MotionBuildResult:
    data = np.load(pose_path)
    fps = float(data["fps"])
    pose_world = data["pose_world"][:, :, :3].astype(np.float64)
    pose_image = data["pose_image"][:, :, :3].astype(np.float64)
    left_hand = data["left_hand_image"].astype(np.float64)
    right_hand = data["right_hand_image"].astype(np.float64)
    left_hand_world = data["left_hand_world"].astype(np.float64)
    right_hand_world = data["right_hand_world"].astype(np.float64)

    world_points = mediapipe_world_to_canonical(pose_world)
    image_points = mediapipe_image_to_canonical(pose_image)
    # For front-facing dictionary signs, image-space arm directions match the visual signing space
    # more reliably than monocular world-depth estimates.
    pose_points = image_points
    pose_points = interpolate_short_gaps(pose_points, max_gap=max(2, int(round(fps * 0.25))))
    left_points = interpolate_short_gaps(mediapipe_image_to_canonical(left_hand), max_gap=max(2, int(round(fps * 0.20))))
    right_points = interpolate_short_gaps(mediapipe_image_to_canonical(right_hand), max_gap=max(2, int(round(fps * 0.20))))
    left_hand_shape = interpolate_short_gaps(mediapipe_world_to_canonical(left_hand_world), max_gap=max(2, int(round(fps * 0.20))))
    right_hand_shape = interpolate_short_gaps(mediapipe_world_to_canonical(right_hand_world), max_gap=max(2, int(round(fps * 0.20))))
    if smoothing:
        pose_points = smooth_landmarks(pose_points, fps=fps, min_cutoff=1.1, beta=0.025)
        left_points = smooth_landmarks(left_points, fps=fps, min_cutoff=1.8, beta=0.01)
        right_points = smooth_landmarks(right_points, fps=fps, min_cutoff=1.8, beta=0.01)
        left_hand_shape = smooth_landmarks(left_hand_shape, fps=fps, min_cutoff=3.0, beta=0.04)
        right_hand_shape = smooth_landmarks(right_hand_shape, fps=fps, min_cutoff=3.0, beta=0.04)
    left_finger_curls = _compute_finger_curls(left_points)
    right_finger_curls = _compute_finger_curls(right_points)
    if smoothing:
        # Finger bend is filtered separately from wrist/palm position so fast
        # handshapes remain responsive while landmark noise cannot flip joints.
        left_finger_curls = smooth_landmarks(left_finger_curls, fps=fps, min_cutoff=3.0, beta=0.04)
        right_finger_curls = smooth_landmarks(right_finger_curls, fps=fps, min_cutoff=3.0, beta=0.04)
    left_finger_curls = _hold_last_finger_curls(left_finger_curls)
    right_finger_curls = _hold_last_finger_curls(right_finger_curls)
    left_finger_curls = _limit_finger_curl_steps(left_finger_curls, max_delta=np.radians(35.0))
    right_finger_curls = _limit_finger_curl_steps(right_finger_curls, max_delta=np.radians(35.0))
    left_finger_directions, left_finger_direction_valid = _build_finger_directions(left_hand_shape)
    right_finger_directions, right_finger_direction_valid = _build_finger_directions(right_hand_shape)
    # Hand landmark directions are noisier than palm translation. Limit the
    # per-frame turn in local palm space before Blender turns them into bone
    # rotations, otherwise a single landmark outlier can spin a finger.
    finger_directions = np.stack([left_finger_directions, right_finger_directions], axis=1)
    finger_direction_valid = np.stack([left_finger_direction_valid, right_finger_direction_valid], axis=1)
    finger_directions = _limit_finger_direction_steps(finger_directions, finger_direction_valid, max_delta=np.radians(24.0))
    finger_directions, finger_direction_constraint_valid = _hold_finger_directions_for_release(
        finger_directions,
        finger_direction_valid,
        release_frames=max(4, int(round(fps * 0.18))),
    )
    finger_direction_influence = _ramp_finger_direction_activation(
        finger_direction_valid,
        finger_direction_constraint_valid,
        ramp_frames=max(4, int(round(fps * 0.18))),
    )

    profile = load_avatar_profile(profile_path)
    bone_map = load_bone_map(bone_map_path)
    rest = RestPose.from_profile(profile)
    bone_names = list(bone_map.keys())
    frame_count = int(data["frame_count"])
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (frame_count, len(bone_names), 1))

    for frame_index in range(frame_count):
        frame_pose = pose_points[frame_index]
        _solve_body_frame(rotations[frame_index], bone_names, bone_map, rest, frame_pose, body_only)
        if not body_only and include_palms:
            _solve_palm_frame(rotations[frame_index], bone_names, bone_map, rest, frame_pose, left_points[frame_index], "Left")
            _solve_palm_frame(rotations[frame_index], bone_names, bone_map, rest, frame_pose, right_points[frame_index], "Right")
        if not body_only and include_fingers:
            _solve_finger_frame(rotations[frame_index], bone_names, left_finger_curls[frame_index], "Left")
            _solve_finger_frame(rotations[frame_index], bone_names, right_finger_curls[frame_index], "Right")

    rotations = enforce_quaternion_continuity(rotations)
    validate_quaternions(rotations)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_translation = np.zeros((frame_count, 3), dtype=np.float32)
    np.savez_compressed(
        output,
        fps=np.array(fps, dtype=np.float32),
        frame_count=np.array(frame_count, dtype=np.int32),
        bone_names=np.array(bone_names, dtype=np.str_),
        avatar_bone_names=np.array([bone_map[name] for name in bone_names], dtype=np.str_),
        ik_target_names=np.array(["LeftWrist", "LeftElbow", "RightWrist", "RightElbow"], dtype=np.str_),
        ik_targets=_build_ik_targets(data["pose_image"][:, :, :3].astype(np.float64), rest, bone_map).astype(np.float32),
        finger_direction_basis=np.array("PALM_ACROSS_FORWARD_NORMAL", dtype=np.str_),
        finger_directions=finger_directions.astype(np.float32),
        finger_direction_valid=finger_direction_valid.astype(np.uint8),
        finger_direction_constraint_valid=finger_direction_constraint_valid.astype(np.uint8),
        finger_direction_influence=finger_direction_influence.astype(np.float32),
        root_translation=root_translation,
        rotations=rotations.astype(np.float32),
        quaternion_format=np.array("WXYZ", dtype=np.str_),
        action_name=np.array(gloss.upper(), dtype=np.str_),
        start_pose=rotations[0].astype(np.float32),
        end_pose=rotations[-1].astype(np.float32),
        solver_notes=np.array(
            [
                "V1 direct solver; avatar FBX bone lengths remain authoritative.",
                "Body and hand landmarks are stored and solved separately.",
                "Finger joints use locally applied, anatomically bounded curl rotations.",
                "Finger direction targets use local MediaPipe hand-world geometry, not global hand placement.",
            ],
            dtype=np.str_,
        ),
    )
    return MotionBuildResult(
        motion_path=output,
        metadata={
            "frame_count": frame_count,
            "fps": fps,
            "bone_count": len(bone_names),
            "body_only": body_only,
            "include_palms": include_palms,
            "include_fingers": include_fingers,
        },
    )


class RestPose:
    def __init__(self, bone_info: dict[str, dict]) -> None:
        self.bone_info = bone_info

    @classmethod
    def from_profile(cls, profile: dict) -> "RestPose":
        return cls({bone["name"]: bone for bone in profile["bones"]})

    def direction(self, bone_name: str) -> np.ndarray:
        bone = self.bone_info[bone_name]
        return normalize_vector(np.array(bone["tail_local"], dtype=np.float64) - np.array(bone["head_local"], dtype=np.float64))

    def rotation_matrix(self, bone_name: str) -> np.ndarray:
        matrix = np.array(self.bone_info[bone_name]["matrix_local"], dtype=np.float64)
        return matrix[:3, :3]


def _solve_body_frame(rotations, bone_names, bone_map, rest: RestPose, pose, body_only: bool) -> None:
    for side, indices in (("Left", LEFT), ("Right", RIGHT)):
        shoulder = pose[indices["shoulder"]]
        elbow = pose[indices["elbow"]]
        wrist = pose[indices["wrist"]]
        upper_vector = elbow - shoulder
        forearm_vector = wrist - elbow
        _set_swing(rotations, bone_names, bone_map, rest, f"{side}UpperArm", upper_vector)
        _set_swing(rotations, bone_names, bone_map, rest, f"{side}ForeArm", forearm_vector)
        if body_only:
            _set_identity(rotations, bone_names, f"{side}Hand")

    left_shoulder = pose[LEFT["shoulder"]]
    right_shoulder = pose[RIGHT["shoulder"]]
    left_hip = pose[23]
    right_hip = pose[24]
    if all(np.isfinite(v).all() for v in [left_shoulder, right_shoulder, left_hip, right_hip]):
        shoulder_axis = normalize_vector(right_shoulder - left_shoulder)
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        up_axis = normalize_vector(shoulder_center - hip_center)
        forward_axis = normalize_vector(np.cross(shoulder_axis, up_axis))
        torso_twist = float(np.clip(forward_axis[0], -0.35, 0.35))
        _set_if_present(rotations, bone_names, "Chest", quaternion_from_axis_angle([0, 0, 1], torso_twist * 0.35))


def _solve_palm_frame(rotations, bone_names, bone_map, rest: RestPose, pose, hand, side: str) -> None:
    if not np.isfinite(hand[[0, 5, 9, 17], :3]).all():
        return
    wrist = hand[0]
    middle = hand[9]
    index = hand[5]
    pinky = hand[17]
    palm_forward = middle - wrist
    palm_horizontal = index - pinky if side == "Left" else pinky - index
    normal = normalize_vector(np.cross(palm_horizontal, palm_forward))
    target = normalize_vector(palm_forward + normal * 0.18)
    _set_swing(rotations, bone_names, bone_map, rest, f"{side}Hand", target, strength=0.45, max_angle=np.radians(35.0))


def _compute_finger_curls(hand_points: np.ndarray) -> np.ndarray:
    """Return non-negative MCP/PIP/DIP flexion angles for each tracked hand."""
    curls = np.full((hand_points.shape[0], len(HAND_CHAINS), 3), np.nan, dtype=np.float64)
    for frame_index, hand in enumerate(hand_points):
        for finger_index, (_, chain) in enumerate(HAND_CHAINS.items()):
            points = [0] + chain
            for joint_index in range(3):
                a, b, c = points[joint_index], points[joint_index + 1], points[joint_index + 2]
                if not np.isfinite(hand[[a, b, c], :3]).all():
                    continue
                parent = hand[a] - hand[b]
                child = hand[c] - hand[b]
                parent_length = np.linalg.norm(parent)
                child_length = np.linalg.norm(child)
                if parent_length < 1e-8 or child_length < 1e-8:
                    continue
                cosine = float(np.clip(np.dot(parent, child) / (parent_length * child_length), -1.0, 1.0))
                curls[frame_index, finger_index, joint_index] = np.pi - np.arccos(cosine)
    return curls


def _build_finger_directions(hand_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit finger-segment directions in each frame's palm coordinate system.

    The hand-world landmarks are intentionally used only for *local hand shape*.
    A palm basis makes the directions independent of the hand's global position;
    Blender later attaches them to the tracked body wrist and avatar palm.
    """
    frame_count = hand_points.shape[0]
    directions = np.full((frame_count, len(HAND_CHAINS), 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, len(HAND_CHAINS), 3), dtype=bool)

    for frame_index, hand in enumerate(hand_points):
        if not np.isfinite(hand[[0, 5, 9, 17], :3]).all():
            continue
        across = _unit_or_none(hand[5] - hand[17])
        forward = _unit_or_none(hand[9] - hand[0])
        if across is None or forward is None:
            continue
        normal = _unit_or_none(np.cross(across, forward))
        if normal is None:
            continue
        # Rebuild forward after the cross product to keep the basis orthogonal.
        forward = _unit_or_none(np.cross(normal, across))
        if forward is None:
            continue
        basis = np.stack([across, forward, normal], axis=0)

        for finger_index, chain in enumerate(HAND_CHAINS.values()):
            for segment_index, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
                if not np.isfinite(hand[[start, end], :3]).all():
                    continue
                direction = _unit_or_none(hand[end] - hand[start])
                if direction is None:
                    continue
                directions[frame_index, finger_index, segment_index] = basis @ direction
                valid[frame_index, finger_index, segment_index] = True

    return directions, valid


def _unit_or_none(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        return None
    return vector / length


def _limit_finger_direction_steps(directions: np.ndarray, valid: np.ndarray, max_delta: float) -> np.ndarray:
    """Spherically limit each tracked local finger direction between frames."""
    output = np.asarray(directions, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    for side_index in range(output.shape[1]):
        for finger_index in range(output.shape[2]):
            for segment_index in range(output.shape[3]):
                previous: np.ndarray | None = None
                for frame_index in range(output.shape[0]):
                    if not valid[frame_index, side_index, finger_index, segment_index]:
                        previous = None
                        continue
                    current = _unit_or_none(output[frame_index, side_index, finger_index, segment_index])
                    if current is None:
                        valid[frame_index, side_index, finger_index, segment_index] = False
                        previous = None
                        continue
                    if previous is not None:
                        dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
                        angle = float(np.arccos(dot))
                        if angle > max_delta:
                            current = _slerp_direction(previous, current, max_delta / angle)
                    output[frame_index, side_index, finger_index, segment_index] = current
                    previous = current
    return output


def _slerp_direction(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    first = _unit_or_none(first)
    second = _unit_or_none(second)
    if first is None or second is None:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    amount = float(np.clip(amount, 0.0, 1.0))
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot > 0.9995:
        result = _unit_or_none(first + amount * (second - first))
        return first if result is None else result
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        return first
    result = _unit_or_none((np.sin((1.0 - amount) * angle) / sine) * first + (np.sin(amount * angle) / sine) * second)
    return first if result is None else result


def _hold_finger_directions_for_release(
    directions: np.ndarray,
    source_valid: np.ndarray,
    release_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze the last known handshape briefly while a lost hand fades out."""
    output = np.asarray(directions, dtype=np.float64).copy()
    source_valid = np.asarray(source_valid, dtype=bool)
    constraint_valid = source_valid.copy()
    release_frames = max(1, int(release_frames))
    for side_index in range(output.shape[1]):
        for finger_index in range(output.shape[2]):
            for segment_index in range(output.shape[3]):
                previous: np.ndarray | None = None
                remaining = 0
                for frame_index in range(output.shape[0]):
                    if source_valid[frame_index, side_index, finger_index, segment_index]:
                        previous = output[frame_index, side_index, finger_index, segment_index].copy()
                        remaining = release_frames
                    elif previous is not None and remaining > 0:
                        output[frame_index, side_index, finger_index, segment_index] = previous
                        constraint_valid[frame_index, side_index, finger_index, segment_index] = True
                        remaining -= 1
                    else:
                        previous = None
                        remaining = 0
    return output, constraint_valid


def _ramp_finger_direction_activation(
    source_valid: np.ndarray,
    constraint_valid: np.ndarray,
    ramp_frames: int,
) -> np.ndarray:
    """Fade a re-acquired or lost handshape between direct-curl and target modes."""
    source_valid = np.asarray(source_valid, dtype=bool)
    constraint_valid = np.asarray(constraint_valid, dtype=bool)
    output = np.zeros(source_valid.shape, dtype=np.float64)
    ramp_frames = max(1, int(ramp_frames))
    for side_index in range(source_valid.shape[1]):
        for finger_index in range(source_valid.shape[2]):
            for segment_index in range(source_valid.shape[3]):
                consecutive = 0
                release_frame = 0
                last_influence = 0.0
                for frame_index in range(source_valid.shape[0]):
                    if source_valid[frame_index, side_index, finger_index, segment_index]:
                        consecutive += 1
                        release_frame = 0
                        last_influence = min(1.0, consecutive / ramp_frames)
                        output[frame_index, side_index, finger_index, segment_index] = last_influence
                    elif constraint_valid[frame_index, side_index, finger_index, segment_index]:
                        release_frame += 1
                        output[frame_index, side_index, finger_index, segment_index] = max(0.0, last_influence - release_frame / (ramp_frames + 1))
                    else:
                        consecutive = 0
                        release_frame = 0
                        last_influence = 0.0
    return output


def _solve_finger_frame(rotations, bone_names, curls: np.ndarray, side: str) -> None:
    for finger_index, finger in enumerate(HAND_CHAINS):
        for joint_index in range(3):
            canonical = f"{side}{finger}{joint_index + 1}"
            if canonical not in bone_names:
                continue
            raw_curl = curls[finger_index, joint_index]
            if not np.isfinite(raw_curl):
                continue
            curl = _bounded_finger_curl(finger, joint_index, float(raw_curl))
            # The calibrated Mixamo-like avatar uses local +X as anatomical
            # flexion for both hands. Local rotations keep palm orientation
            # independent from the finger's curl.
            _set_if_present(rotations, bone_names, canonical, quaternion_from_axis_angle([1.0, 0.0, 0.0], curl))


def _bounded_finger_curl(finger: str, joint_index: int, raw_curl: float) -> float:
    if finger == "Thumb":
        limits_degrees = (48.0, 65.0, 55.0)
        scale = 0.72
    else:
        limits_degrees = (70.0, 100.0, 80.0)
        scale = 0.72
    return float(np.clip(raw_curl * scale, 0.0, np.radians(limits_degrees[joint_index])))


def _limit_finger_curl_steps(curls: np.ndarray, max_delta: float) -> np.ndarray:
    """Prevent a detected hand from producing a one-frame finger snap."""
    output = np.asarray(curls, dtype=np.float64).copy()
    for finger_index in range(output.shape[1]):
        for joint_index in range(output.shape[2]):
            previous = np.nan
            for frame_index in range(output.shape[0]):
                current = output[frame_index, finger_index, joint_index]
                if not np.isfinite(current):
                    previous = np.nan
                    continue
                if np.isfinite(previous):
                    current = float(np.clip(current, previous - max_delta, previous + max_delta))
                    output[frame_index, finger_index, joint_index] = current
                else:
                    # A re-acquired hand begins from the unanimated/rest hand
                    # instead of snapping directly from identity to a noisy pose.
                    current = float(np.clip(current, 0.0, max_delta))
                    output[frame_index, finger_index, joint_index] = current
                previous = current
    return output


def _hold_last_finger_curls(curls: np.ndarray) -> np.ndarray:
    """Freeze the last observed handshape during a tracking gap, never inventing new motion."""
    output = np.asarray(curls, dtype=np.float64).copy()
    for finger_index in range(output.shape[1]):
        for joint_index in range(output.shape[2]):
            previous = np.nan
            for frame_index in range(output.shape[0]):
                current = output[frame_index, finger_index, joint_index]
                if np.isfinite(current):
                    previous = current
                elif np.isfinite(previous):
                    output[frame_index, finger_index, joint_index] = previous
    return output


def _set_swing(
    rotations,
    bone_names,
    bone_map,
    rest: RestPose,
    canonical_name: str,
    target_vector,
    strength: float = 0.65,
    max_angle: float = np.radians(75.0),
) -> None:
    if canonical_name not in bone_map or not np.isfinite(target_vector).all():
        return
    avatar_bone = bone_map[canonical_name]
    rest_direction = rest.direction(avatar_bone)
    target = normalize_vector(target_vector)
    swing_armature = quaternion_from_vectors(rest_direction, target)
    rest_rotation = rest.rotation_matrix(avatar_bone)
    local_matrix = rest_rotation.T @ quaternion_to_matrix(swing_armature) @ rest_rotation
    q = quaternion_from_matrix(local_matrix)
    q = limit_quaternion_angle(q, max_angle)
    q = quaternion_slerp(np.array([1.0, 0.0, 0.0, 0.0]), q, strength)
    _set_if_present(rotations, bone_names, canonical_name, q)


def _set_if_present(rotations, bone_names, canonical_name: str, q) -> None:
    if canonical_name in bone_names:
        rotations[bone_names.index(canonical_name)] = normalize_quaternion(q)


def _set_identity(rotations, bone_names, canonical_name: str) -> None:
    _set_if_present(rotations, bone_names, canonical_name, np.array([1.0, 0.0, 0.0, 0.0]))


def _build_ik_targets(pose_image: np.ndarray, rest: RestPose, bone_map: dict[str, str]) -> np.ndarray:
    frame_count = pose_image.shape[0]
    targets = np.zeros((frame_count, 4, 3), dtype=np.float64)
    left_shoulder_rest = np.array(rest.bone_info[bone_map["LeftUpperArm"]]["head_local"], dtype=np.float64)
    right_shoulder_rest = np.array(rest.bone_info[bone_map["RightUpperArm"]]["head_local"], dtype=np.float64)
    hips_rest = np.array(rest.bone_info[bone_map["Hips"]]["head_local"], dtype=np.float64)
    avatar_shoulder_center = (left_shoulder_rest + right_shoulder_rest) * 0.5
    avatar_shoulder_width = abs(left_shoulder_rest[0] - right_shoulder_rest[0])
    avatar_torso_height = abs(avatar_shoulder_center[1] - hips_rest[1])
    rest_depth = float((left_shoulder_rest[2] + right_shoulder_rest[2]) * 0.5)

    previous = None
    for frame_index in range(frame_count):
        frame = pose_image[frame_index]
        if not np.isfinite(frame[[11, 12], :2]).all():
            if previous is not None:
                targets[frame_index] = previous
            continue
        source_left_shoulder = frame[11]
        source_right_shoulder = frame[12]
        source_left_hip = frame[23]
        source_right_hip = frame[24]
        source_center_x = float((source_left_shoulder[0] + source_right_shoulder[0]) * 0.5)
        source_center_y = float((source_left_shoulder[1] + source_right_shoulder[1]) * 0.5)
        source_width = abs(float(source_left_shoulder[0] - source_right_shoulder[0]))
        source_hip_center_y = float((source_left_hip[1] + source_right_hip[1]) * 0.5) if np.isfinite(frame[[23, 24], 1]).all() else source_center_y + 0.32
        source_torso_height = abs(source_hip_center_y - source_center_y)
        scale_x = avatar_shoulder_width / max(source_width, 0.05)
        scale_y = avatar_torso_height / max(source_torso_height, 0.18)

        mapped = []
        for landmark_index in [15, 13, 16, 14]:
            landmark = frame[landmark_index]
            if not np.isfinite(landmark[:2]).all():
                mapped.append(previous[len(mapped)] if previous is not None else avatar_shoulder_center)
                continue
            x = avatar_shoulder_center[0] + (float(landmark[0]) - source_center_x) * scale_x
            y = avatar_shoulder_center[1] + (source_center_y - float(landmark[1])) * scale_y
            z = rest_depth + 16.0
            mapped.append(np.array([x, y, z], dtype=np.float64))
        targets[frame_index] = np.array(mapped, dtype=np.float64)
        previous = targets[frame_index].copy()
    return targets
