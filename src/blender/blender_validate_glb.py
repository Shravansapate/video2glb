from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def main() -> int:
    args = parse_args()
    result = validate(args.glb, args.motion, args.profile, args.bone_map)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] == "FAIL":
        raise RuntimeError(result["reasons"])
    return 0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bone-map", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def validate(glb: str, motion: str, profile: str, bone_map: str) -> dict:
    reasons: list[str] = []
    review_reasons: list[str] = []
    path = Path(glb)
    if not path.exists() or path.stat().st_size <= 0:
        return {"status": "FAIL", "reasons": ["GLB is missing or empty."]}

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [obj.name for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    actions = list(bpy.data.actions)
    motion_data = np.load(motion)
    expected_bones = [str(name) for name in motion_data["avatar_bone_names"]]

    if not meshes:
        reasons.append("No mesh found after GLB import.")
    if not armatures:
        reasons.append("No armature found after GLB import.")
    if len(actions) != 1:
        reasons.append(f"Expected exactly one animation, found {len(actions)}.")
    if armatures:
        armature = armatures[0]
        existing = set(armature.pose.bones.keys())
        missing = [bone for bone in expected_bones if bone not in existing]
        required_arms = [bone for bone in expected_bones if any(token in bone.lower() for token in ["arm", "forearm", "hand"])]
        if missing:
            reasons.append(f"Missing animated bones: {missing[:10]}")
        if not required_arms:
            reasons.append("No required arm/hand bones present in motion map.")
        hand_visibility = _hand_visibility_metrics(armature, motion_data)
        if hand_visibility["status"] != "PASS":
            reasons.extend(hand_visibility["reasons"])
        finger_motion = _finger_motion_metrics(armature, motion_data)
    if finger_motion["status"] != "PASS":
        if _has_structural_motion_error(finger_motion):
            reasons.extend(finger_motion["reasons"])
        else:
            review_reasons.extend(finger_motion["reasons"])
        finger_retargeting = _finger_target_metrics(armature, motion_data)
        if finger_retargeting["status"] != "PASS":
            review_reasons.extend(finger_retargeting["reasons"])
        retargeting = _wrist_target_metrics(armature, motion_data, _load_profile(profile))
        if retargeting["status"] != "PASS":
            review_reasons.extend(retargeting["reasons"])
    else:
        hand_visibility = {"status": "FAIL", "reasons": ["No armature for hand visibility check."]}
        finger_motion = {"status": "FAIL", "reasons": ["No armature for finger-motion validation."]}
        finger_retargeting = {"status": "FAIL", "reasons": ["No armature for finger-target validation."]}
        retargeting = {"status": "FAIL", "reasons": ["No armature for wrist-target validation."]}

    rotations = motion_data["rotations"]
    if not np.isfinite(rotations).all():
        reasons.append("Motion file contains NaN or Infinity.")
    if np.max(np.abs(np.linalg.norm(rotations, axis=2) - 1.0)) > 1e-3:
        reasons.append("Motion quaternions are not unit length.")

    frame_count = int(motion_data["frame_count"])
    source_fps = float(motion_data["fps"])
    expected_duration = (frame_count - 1) / source_fps if source_fps else 0.0
    actual_duration = 0.0
    imported_timeline_fps = float(bpy.context.scene.render.fps) / max(float(bpy.context.scene.render.fps_base), 1e-8)
    if actions:
        start, end = actions[0].frame_range
        # glTF stores keyframe times in seconds. On import Blender maps those
        # seconds onto the current scene timeline (normally 24 FPS), which is
        # independent from the source video's FPS. Dividing by source_fps here
        # falsely rejects every valid 25/29.97-FPS GLB as too short.
        actual_duration = (end - start) / imported_timeline_fps if imported_timeline_fps else 0.0
        if abs(actual_duration - expected_duration) > max(1.0 / source_fps, 0.05):
            reasons.append("Animation duration does not match motion duration.")

    status = "FAIL" if reasons else ("REVIEW" if review_reasons else "PASS")
    return {
        "status": status,
        "reasons": reasons,
        "review_reasons": review_reasons,
        "mesh_count": len(meshes),
        "armature_count": len(armatures),
        "animation_count": len(actions),
        "expected_duration_seconds": expected_duration,
        "actual_duration_seconds": actual_duration,
        "source_fps": source_fps,
        "imported_timeline_fps": imported_timeline_fps,
        "file_size": path.stat().st_size,
        "hand_visibility": hand_visibility,
        "finger_motion": finger_motion,
        "finger_retargeting": finger_retargeting,
        "retargeting": retargeting,
    }


def _load_profile(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(bone["name"]): bone for bone in payload.get("bones", [])}


def _has_structural_motion_error(metric: dict) -> bool:
    """Keep malformed animation data as FAIL; route quality concerns to REVIEW."""
    structural_markers = ("no finger rotation channels", "missing finger bones", "zero-length", "non-finite")
    return any(marker in reason.lower() for reason in metric.get("reasons", []) for marker in structural_markers)


def _wrist_target_metrics(armature, motion_data, profile: dict[str, dict]) -> dict:
    if "ik_targets" not in motion_data.files:
        return {
            "status": "REVIEW",
            "reasons": ["Motion has no wrist targets, so endpoint retargeting could not be checked."],
        }

    canonical_names = [str(name) for name in motion_data["bone_names"]]
    avatar_names = [str(name) for name in motion_data["avatar_bone_names"]]
    bone_map = dict(zip(canonical_names, avatar_names))
    required = ["LeftUpperArm", "LeftForeArm", "LeftHand", "RightUpperArm", "RightForeArm", "RightHand"]
    missing = [name for name in required if bone_map.get(name) not in armature.pose.bones or bone_map.get(name) not in profile]
    if missing:
        return {"status": "FAIL", "reasons": [f"Cannot validate wrist targets; missing mapped bones: {missing}"]}

    targets = motion_data["ik_targets"].astype(float)
    frame_count = min(int(motion_data["frame_count"]), len(targets))
    pairs = [
        ("LeftUpperArm", "LeftForeArm", "LeftHand", 0),
        ("RightUpperArm", "RightForeArm", "RightHand", 2),
    ]
    object_scale = float(np.mean(np.abs(armature.matrix_world.to_scale())))
    errors: list[float] = []
    for frame_index in range(frame_count):
        bpy.context.scene.frame_set(frame_index + 1)
        bpy.context.view_layer.update()
        for upper_name, fore_name, hand_name, target_index in pairs:
            target_local = targets[frame_index, target_index]
            if not np.isfinite(target_local).all():
                continue
            upper = bone_map[upper_name]
            fore = bone_map[fore_name]
            hand = bone_map[hand_name]
            chain_length = (float(profile[upper]["length"]) + float(profile[fore]["length"])) * object_scale
            if chain_length <= 1e-6:
                continue
            target_world = armature.matrix_world @ Vector(target_local)
            actual_world = armature.matrix_world @ armature.pose.bones[hand].head
            errors.append(float((actual_world - target_world).length / chain_length))

    if not errors:
        return {"status": "FAIL", "reasons": ["No finite wrist targets were available for retarget validation."]}

    values = np.asarray(errors, dtype=np.float64)
    mean_error = float(np.mean(values))
    p95_error = float(np.percentile(values, 95))
    max_error = float(np.max(values))
    reasons: list[str] = []
    if mean_error > 0.20 or p95_error > 0.35:
        status = "FAIL"
        reasons.append("Avatar wrist paths deviate substantially from the solved source wrist targets.")
    elif mean_error > 0.08 or p95_error > 0.15:
        status = "REVIEW"
        reasons.append("Avatar wrist paths need review because endpoint tracking is not close enough.")
    else:
        status = "PASS"

    return {
        "status": status,
        "reasons": reasons,
        "sample_count": len(errors),
        "mean_normalized_wrist_error": mean_error,
        "p95_normalized_wrist_error": p95_error,
        "max_normalized_wrist_error": max_error,
    }


def _finger_motion_metrics(armature, motion_data) -> dict:
    canonical_names = [str(name) for name in motion_data["bone_names"]]
    avatar_names = [str(name) for name in motion_data["avatar_bone_names"]]
    finger_pairs = [
        (canonical, avatar)
        for canonical, avatar in zip(canonical_names, avatar_names)
        if any(finger in canonical for finger in ["Thumb", "Index", "Middle", "Ring", "Little"])
    ]
    if not finger_pairs:
        return {"status": "FAIL", "reasons": ["Motion file has no finger rotation channels."]}
    missing = [canonical for canonical, avatar in finger_pairs if avatar not in armature.pose.bones]
    if missing:
        return {"status": "FAIL", "reasons": [f"GLB is missing finger bones: {missing[:10]}"]}

    if "finger_direction_valid" in motion_data.files:
        return _finger_local_direction_motion_metrics(armature, motion_data, finger_pairs)

    frame_count = int(motion_data["frame_count"])
    directions = np.empty((frame_count, len(finger_pairs), 3), dtype=np.float64)
    for frame in range(frame_count):
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for index, (_, avatar) in enumerate(finger_pairs):
            bone = armature.pose.bones[avatar]
            direction = (armature.matrix_world @ bone.tail) - (armature.matrix_world @ bone.head)
            if direction.length < 1e-8:
                return {"status": "FAIL", "reasons": [f"Baked GLB has a zero-length finger direction for {avatar}."]}
            directions[frame, index] = direction.normalized()
    if not np.isfinite(directions).all():
        return {"status": "FAIL", "reasons": ["Baked GLB has non-finite finger directions."]}
    if len(directions) < 2:
        return {"status": "REVIEW", "reasons": ["Motion is too short to measure finger angular continuity."]}
    dots = np.sum(directions[1:] * directions[:-1], axis=2)
    deltas = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    max_delta = float(np.max(deltas))
    p99_delta = float(np.percentile(deltas, 99))
    max_frame_index, max_bone_index = np.unravel_index(int(np.argmax(deltas)), deltas.shape)
    reasons: list[str] = []
    if max_delta > 60.0 or p99_delta > 40.0:
        status = "FAIL"
        reasons.append("Visible finger direction contains an implausibly large frame-to-frame spike.")
    elif max_delta > 35.0 or p99_delta > 25.0:
        status = "REVIEW"
        reasons.append("Visible finger direction has a sharp transition that should be reviewed.")
    else:
        status = "PASS"
    return {
        "status": status,
        "reasons": reasons,
        "finger_channel_count": len(finger_pairs),
        "max_visible_direction_delta_degrees": max_delta,
        "p99_visible_direction_delta_degrees": p99_delta,
        "max_delta_frame": int(max_frame_index + 2),
        "max_delta_bone": finger_pairs[int(max_bone_index)][0],
    }


def _finger_local_direction_motion_metrics(armature, motion_data, finger_pairs: list[tuple[str, str]]) -> dict:
    """Check visible finger turns relative to the animated palm, not world space."""
    fingers = ("Thumb", "Index", "Middle", "Ring", "Little")
    pair_index = {canonical: index for index, (canonical, _) in enumerate(finger_pairs)}
    avatar_map = dict(finger_pairs)
    full_bone_map = dict(zip([str(name) for name in motion_data["bone_names"]], [str(name) for name in motion_data["avatar_bone_names"]]))
    valid = motion_data["finger_direction_valid"].astype(bool)
    influence = motion_data["finger_direction_influence"].astype(float) if "finger_direction_influence" in motion_data.files else valid.astype(float)
    previous: list[tuple[int, np.ndarray] | None] = [None] * len(finger_pairs)
    deltas: list[tuple[float, int, str]] = []
    frame_count = min(int(motion_data["frame_count"]), len(valid))

    for frame in range(frame_count):
        bpy.context.scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for side_index, side in enumerate(("Left", "Right")):
            hand_name = full_bone_map.get(f"{side}Hand", "")
            if hand_name not in armature.pose.bones:
                continue
            hand_inverse = (armature.matrix_world @ armature.pose.bones[hand_name].matrix).to_3x3().inverted()
            for finger_index, finger in enumerate(fingers):
                for segment_index in range(3):
                    canonical = f"{side}{finger}{segment_index + 1}"
                    if canonical not in pair_index:
                        continue
                    index = pair_index[canonical]
                    is_active = bool(valid[frame, side_index, finger_index, segment_index]) and float(influence[frame, side_index, finger_index, segment_index]) >= 0.999
                    if not is_active:
                        previous[index] = None
                        continue
                    pose_bone = armature.pose.bones[avatar_map[canonical]]
                    world_direction = (armature.matrix_world @ pose_bone.tail) - (armature.matrix_world @ pose_bone.head)
                    local_direction = hand_inverse @ world_direction
                    if local_direction.length < 1e-8:
                        previous[index] = None
                        continue
                    local_direction.normalize()
                    prior = previous[index]
                    if prior is not None and prior[0] == frame - 1:
                        dot = max(-1.0, min(1.0, prior[1].dot(local_direction)))
                        deltas.append((float(np.degrees(np.arccos(dot))), frame + 1, canonical))
                    previous[index] = (frame, local_direction)

    if not deltas:
        return {"status": "REVIEW", "reasons": ["No consecutive, fully tracked finger samples were available for local continuity validation."]}
    values = np.asarray([delta for delta, _, _ in deltas], dtype=np.float64)
    max_index = int(np.argmax(values))
    max_delta, max_frame, max_bone = deltas[max_index]
    p99_delta = float(np.percentile(values, 99))
    reasons: list[str] = []
    if max_delta > 45.0 or p99_delta > 32.0:
        status = "FAIL"
        reasons.append("Finger shape contains an implausibly large turn relative to its palm.")
    elif max_delta > 30.0 or p99_delta > 26.0:
        status = "REVIEW"
        reasons.append("Finger shape has a sharp local transition that should be reviewed.")
    else:
        status = "PASS"
    return {
        "status": status,
        "reasons": reasons,
        "finger_channel_count": len(finger_pairs),
        "sample_count": len(deltas),
        "max_local_direction_delta_degrees": float(max_delta),
        "p99_local_direction_delta_degrees": p99_delta,
        "max_delta_frame": int(max_frame),
        "max_delta_bone": max_bone,
    }


def _finger_target_metrics(armature, motion_data) -> dict:
    if "finger_directions" not in motion_data.files or "finger_direction_valid" not in motion_data.files:
        return {
            "status": "REVIEW",
            "reasons": ["Motion has no local hand-world finger directions for source-to-avatar validation."],
        }

    canonical_names = [str(name) for name in motion_data["bone_names"]]
    avatar_names = [str(name) for name in motion_data["avatar_bone_names"]]
    bone_map = dict(zip(canonical_names, avatar_names))
    fingers = ("Thumb", "Index", "Middle", "Ring", "Little")
    directions = motion_data["finger_directions"].astype(np.float64)
    valid = motion_data["finger_direction_valid"].astype(bool)
    errors: list[float] = []

    for side_index, side in enumerate(("Left", "Right")):
        hand_name = bone_map.get(f"{side}Hand", "")
        if hand_name not in armature.pose.bones:
            return {"status": "FAIL", "reasons": [f"Missing mapped {side} hand for finger-target validation."]}
        axes_local = _hand_rest_palm_axes_local(armature, hand_name, bone_map, side)
        if axes_local is None:
            return {"status": "FAIL", "reasons": [f"Could not derive a rest palm basis for {side}."]}
        for frame in range(min(int(motion_data["frame_count"]), len(directions))):
            bpy.context.scene.frame_set(frame + 1)
            bpy.context.view_layer.update()
            hand_world = armature.matrix_world @ armature.pose.bones[hand_name].matrix
            for finger_index, finger in enumerate(fingers):
                for segment_index in range(3):
                    if not valid[frame, side_index, finger_index, segment_index]:
                        continue
                    bone_name = bone_map.get(f"{side}{finger}{segment_index + 1}", "")
                    if bone_name not in armature.pose.bones:
                        continue
                    components = directions[frame, side_index, finger_index, segment_index]
                    expected_local = sum((axes_local[axis] * float(components[axis]) for axis in range(3)), Vector((0.0, 0.0, 0.0)))
                    expected_world = hand_world.to_3x3() @ expected_local
                    pose_bone = armature.pose.bones[bone_name]
                    actual_world = (armature.matrix_world @ pose_bone.tail) - (armature.matrix_world @ pose_bone.head)
                    if expected_world.length < 1e-8 or actual_world.length < 1e-8:
                        continue
                    dot = max(-1.0, min(1.0, expected_world.normalized().dot(actual_world.normalized())))
                    errors.append(float(np.degrees(np.arccos(dot))))

    if not errors:
        return {"status": "FAIL", "reasons": ["No valid finger directions were available for source-to-avatar validation."]}
    values = np.asarray(errors, dtype=np.float64)
    mean_error = float(np.mean(values))
    p95_error = float(np.percentile(values, 95))
    reasons: list[str] = []
    if mean_error > 30.0 or p95_error > 55.0:
        status = "FAIL"
        reasons.append("Baked finger directions deviate substantially from the local source hand geometry.")
    elif mean_error > 15.0 or p95_error > 30.0:
        status = "REVIEW"
        reasons.append("Baked finger directions should be visually reviewed against the source handshape.")
    else:
        status = "PASS"
    return {
        "status": status,
        "reasons": reasons,
        "sample_count": len(errors),
        "mean_direction_error_degrees": mean_error,
        "p95_direction_error_degrees": p95_error,
    }


def _hand_rest_palm_axes_local(armature, hand_name: str, bone_map: dict[str, str], side: str):
    hand = armature.data.bones[hand_name]
    index = armature.data.bones[bone_map[f"{side}Index1"]]
    middle = armature.data.bones[bone_map[f"{side}Middle1"]]
    little = armature.data.bones[bone_map[f"{side}Little1"]]
    inverse = hand.matrix_local.inverted()
    wrist_local = inverse @ hand.head_local
    index_local = inverse @ index.head_local
    middle_local = inverse @ middle.head_local
    little_local = inverse @ little.head_local
    across = _vector_or_none(index_local - little_local)
    forward = _vector_or_none(middle_local - wrist_local)
    if across is None or forward is None:
        return None
    normal = _vector_or_none(across.cross(forward))
    if normal is None:
        return None
    forward = _vector_or_none(normal.cross(across))
    if forward is None:
        return None
    return (across, forward, normal)


def _vector_or_none(vector):
    result = Vector(vector)
    if not np.isfinite(result[:]).all() or result.length < 1e-8:
        return None
    return result.normalized()


def _hand_visibility_metrics(armature, motion_data) -> dict:
    canonical_names = [str(name) for name in motion_data["bone_names"]]
    avatar_names = [str(name) for name in motion_data["avatar_bone_names"]]
    bone_map = dict(zip(canonical_names, avatar_names))
    required = ["LeftHand", "RightHand", "LeftIndex1", "RightIndex1", "LeftThumb1", "RightThumb1"]
    missing = [name for name in required if bone_map.get(name) not in armature.pose.bones]
    if missing:
        return {"status": "FAIL", "reasons": [f"Required visible hand/finger bones missing: {missing}"]}

    left_hand = bone_map["LeftHand"]
    right_hand = bone_map["RightHand"]
    chest = bone_map.get("Chest") if bone_map.get("Chest") in armature.pose.bones else None
    frame_count = int(motion_data["frame_count"])
    samples = sorted(set([1, max(1, frame_count // 5), max(1, frame_count // 2), max(1, 4 * frame_count // 5), frame_count]))
    visible_samples = 0
    distances = []
    for frame in samples:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        chest_x = 0.0
        if chest:
            chest_x = float((armature.matrix_world @ armature.pose.bones[chest].head).x)
        left_x = float((armature.matrix_world @ armature.pose.bones[left_hand].head).x)
        right_x = float((armature.matrix_world @ armature.pose.bones[right_hand].head).x)
        distance = abs(left_x - chest_x) + abs(right_x - chest_x)
        distances.append(distance)
        if distance > 0.35:
            visible_samples += 1

    action = bpy.data.actions[0] if bpy.data.actions else None
    animated_finger_bones = 0
    if action:
        for canonical_name in required[2:]:
            avatar_name = bone_map[canonical_name]
            if any(avatar_name in fcurve.data_path for fcurve in action.fcurves):
                animated_finger_bones += 1

    reasons = []
    if visible_samples < max(2, len(samples) // 2):
        reasons.append("Hand bones appear collapsed into or hidden by the torso in sampled frames.")
    if animated_finger_bones < 4:
        reasons.append("Finger animation channels are missing or too sparse.")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "sampled_frames": samples,
        "visible_samples": visible_samples,
        "hand_separation_distances": distances,
        "animated_finger_bones": animated_finger_bones,
    }


if __name__ == "__main__":
    raise SystemExit(main())
