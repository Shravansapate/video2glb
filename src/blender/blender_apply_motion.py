from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.blender.blender_utils import (  # noqa: E402
    export_avatar_glb,
    import_avatar,
    isolate_target_animation,
    purge_orphans,
    reset_scene,
    validate_animation_state,
)


FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


def main() -> int:
    args = parse_args()
    reset_scene()
    purge_orphans()
    armature = import_avatar(args.avatar)
    motion = np.load(args.motion)
    fps = float(motion["fps"])
    frame_count = int(motion["frame_count"])
    avatar_bone_names = [str(name) for name in motion["avatar_bone_names"]]
    rotations = motion["rotations"].astype(float)
    root_translation = motion["root_translation"].astype(float)
    action_name = str(motion["action_name"])
    ik_targets = motion["ik_targets"].astype(float) if args.use_ik and "ik_targets" in motion.files else None
    finger_directions = motion["finger_directions"].astype(float) if args.use_finger_tracking and "finger_directions" in motion.files else None
    finger_direction_valid = motion["finger_direction_constraint_valid"].astype(bool) if finger_directions is not None and "finger_direction_constraint_valid" in motion.files else None
    if finger_direction_valid is None and finger_directions is not None and "finger_direction_valid" in motion.files:
        finger_direction_valid = motion["finger_direction_valid"].astype(bool)
    finger_direction_influence = motion["finger_direction_influence"].astype(float) if finger_directions is not None and "finger_direction_influence" in motion.files else None

    bpy.context.scene.render.fps = int(round(fps))
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame_count
    action = bpy.data.actions.new(action_name)
    armature.animation_data_create()
    armature.animation_data.action = action
    bone_map = {
        str(canonical): str(avatar)
        for canonical, avatar in zip(motion["bone_names"], motion["avatar_bone_names"])
    }
    ik_objects, ik_constraints = setup_arm_ik(armature, ik_targets, bone_map) if ik_targets is not None else ({}, {})
    ik_calibration = calibrate_arm_poles(armature, ik_objects, ik_constraints, ik_targets) if ik_targets is not None else {}
    finger_objects, finger_constraints = setup_finger_tracking(armature, finger_directions, bone_map) if finger_directions is not None else ({}, {})

    for frame_index in range(frame_count):
        blender_frame = frame_index + 1
        bpy.context.scene.frame_set(blender_frame)
        if ik_targets is not None:
            update_ik_targets(armature, ik_objects, ik_targets[frame_index], blender_frame)
        for bone_index, bone_name in enumerate(avatar_bone_names):
            if bone_name not in armature.pose.bones:
                continue
            if ik_targets is not None and any(part in bone_name for part in ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm"]):
                continue
            pose_bone = armature.pose.bones[bone_name]
            pose_bone.rotation_mode = "QUATERNION"
            w, x, y, z = rotations[frame_index, bone_index]
            if not np.isfinite([w, x, y, z]).all():
                raise RuntimeError(f"Non-finite quaternion for {bone_name} frame {frame_index}.")
            pose_bone.rotation_quaternion = Quaternion((float(w), float(x), float(y), float(z)))
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)

        if args.apply_root_translation:
            armature.location = Vector(root_translation[frame_index])
            armature.keyframe_insert(data_path="location", frame=blender_frame)
        bpy.context.view_layer.update()
        if finger_directions is not None:
            update_finger_targets(
                armature,
                finger_objects,
                finger_constraints,
                finger_directions[frame_index],
                finger_direction_valid[frame_index] if finger_direction_valid is not None else None,
                finger_direction_influence[frame_index] if finger_direction_influence is not None else None,
                blender_frame,
            )

    bpy.context.view_layer.update()
    if ik_targets is not None or finger_directions is not None:
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.nla.bake(
            frame_start=1,
            frame_end=frame_count,
            only_selected=False,
            visual_keying=True,
            clear_constraints=True,
            clear_parents=False,
            use_current_action=True,
            bake_types={"POSE"},
        )
        for obj in [*ik_objects.values(), *finger_objects.values()]:
            bpy.data.objects.remove(obj, do_unlink=True)
    isolate_target_animation(armature, action)
    validate_animation_state(armature, avatar_bone_names)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_avatar_glb(args.output, armature)
    if args.ik_report:
        Path(args.ik_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ik_report).write_text(json.dumps(ik_calibration, indent=2), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bone-map", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply-root-translation", action="store_true")
    parser.add_argument("--use-ik", action="store_true", help="Use body-relative wrist targets with avatar-length arm IK.")
    parser.add_argument("--use-finger-tracking", action="store_true", help="Use local hand-world finger directions with constrained Blender tracking.")
    parser.add_argument("--ik-report", help="Optional JSON report containing per-clip arm IK pole calibration.")
    return parser.parse_args(argv)


def setup_arm_ik(armature, ik_targets, bone_map: dict[str, str]):
    objects = {}
    constraints = {}
    if ik_targets is None:
        return objects, constraints
    for name in ["LeftWrist", "LeftElbow", "RightWrist", "RightElbow"]:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        empty = bpy.context.object
        empty.name = f"IK_{name}"
        empty.empty_display_size = 3.0
        objects[name] = empty

    left_forearm = bone_map.get("LeftForeArm", "")
    left_constraint = add_ik_constraint(
        armature,
        left_forearm,
        objects["LeftWrist"],
        objects["LeftElbow"],
        pole_angle=0.0,
    )
    constraints["Left"] = {"constraint": left_constraint, "forearm_name": left_forearm, "elbow_target_index": 1}
    right_forearm = bone_map.get("RightForeArm", "")
    right_constraint = add_ik_constraint(
        armature,
        right_forearm,
        objects["RightWrist"],
        objects["RightElbow"],
        pole_angle=0.0,
    )
    constraints["Right"] = {"constraint": right_constraint, "forearm_name": right_forearm, "elbow_target_index": 3}
    return objects, constraints


def add_ik_constraint(armature, bone_name: str, target, pole, pole_angle: float):
    if bone_name not in armature.pose.bones:
        return None
    pose_bone = armature.pose.bones[bone_name]
    constraint = pose_bone.constraints.new(type="IK")
    constraint.name = "VideoIK"
    constraint.target = target
    constraint.pole_target = pole
    constraint.pole_angle = pole_angle
    constraint.chain_count = 2
    # The target represents the wrist position.  The tail of a forearm is the
    # wrist; its head is the elbow.  Solving the head collapses the hand into
    # the torso, so the IK end effector must be the tail.
    constraint.use_tail = True
    constraint.use_rotation = False
    return constraint


def calibrate_arm_poles(armature, objects: dict, constraints: dict, targets: np.ndarray) -> dict:
    """Find the rig pole angles that keep elbows closest to tracked elbows."""
    if not constraints:
        return {}

    sample_count = min(72, len(targets))
    sample_indices = np.unique(np.linspace(0, len(targets) - 1, sample_count, dtype=int))
    candidate_angles = np.linspace(-np.pi, np.pi, 33)
    result: dict[str, dict] = {}

    for side, details in constraints.items():
        constraint = details["constraint"]
        forearm_name = details["forearm_name"]
        elbow_target_index = details["elbow_target_index"]
        if constraint is None:
            continue
        if forearm_name not in armature.pose.bones:
            continue
        candidate_scores: list[tuple[float, float]] = []
        for angle in candidate_angles:
            constraint.pole_angle = float(angle)
            errors: list[float] = []
            for frame_index in sample_indices:
                bpy.context.scene.frame_set(int(frame_index) + 1)
                _set_ik_target_locations(armature, objects, targets[frame_index])
                bpy.context.view_layer.update()
                actual = armature.matrix_world @ armature.pose.bones[forearm_name].head
                expected = armature.matrix_world @ Vector(targets[frame_index, elbow_target_index])
                errors.append(float((actual - expected).length))
            candidate_scores.append((float(np.mean(errors)), float(angle)))

        mean_error, best_angle = min(candidate_scores, key=lambda item: item[0])
        constraint.pole_angle = best_angle
        result[side] = {
            "pole_angle_radians": best_angle,
            "sample_count": int(len(sample_indices)),
            "mean_elbow_target_error": mean_error,
        }

    bpy.context.scene.frame_set(1)
    _set_ik_target_locations(armature, objects, targets[0])
    bpy.context.view_layer.update()
    return result


def update_ik_targets(armature, objects: dict, targets, frame: int) -> None:
    _set_ik_target_locations(armature, objects, targets)
    for obj in objects.values():
        obj.keyframe_insert(data_path="location", frame=frame)


def _set_ik_target_locations(armature, objects: dict, targets) -> None:
    mapping = {
        "LeftWrist": 0,
        "LeftElbow": 1,
        "RightWrist": 2,
        "RightElbow": 3,
    }
    for name, index in mapping.items():
        obj = objects[name]
        obj.location = armature.matrix_world @ Vector(targets[index])


def setup_finger_tracking(armature, finger_directions: np.ndarray, bone_map: dict[str, str]):
    """Create one local-direction target per finger segment.

    The targets are evaluated in the animated hand's local palm frame, so the
    MediaPipe hand-world data supplies shape only and never becomes a global
    body coordinate system.
    """
    objects: dict[str, bpy.types.Object] = {}
    constraints: dict[str, dict] = {}
    if finger_directions is None:
        return objects, constraints

    for side in ("Left", "Right"):
        hand_name = bone_map.get(f"{side}Hand", "")
        canonical_finger_names = [f"{side}{finger}{segment}" for finger in FINGERS for segment in range(1, 4)]
        if hand_name not in armature.pose.bones or any(bone_map.get(name, "") not in armature.pose.bones for name in canonical_finger_names):
            continue
        axes_local = _hand_rest_palm_axes_local(armature, hand_name, bone_map, side)
        if axes_local is None:
            continue
        side_constraints: dict[str, dict] = {"hand_name": hand_name, "axes_local": axes_local, "fingers": {}}
        for finger_index, finger in enumerate(FINGERS):
            segments: list[dict] = []
            for segment_index in range(3):
                canonical = f"{side}{finger}{segment_index + 1}"
                bone_name = bone_map[canonical]
                bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
                target = bpy.context.object
                target.name = f"FingerTarget_{side}_{finger}_{segment_index + 1}"
                target.empty_display_size = 0.5
                objects[target.name] = target
                constraint = armature.pose.bones[bone_name].constraints.new(type="DAMPED_TRACK")
                constraint.name = "VideoFingerDirection"
                constraint.target = target
                constraint.track_axis = "TRACK_Y"
                constraint.influence = 0.0
                segments.append({"target": target, "constraint": constraint, "bone_name": bone_name})
            side_constraints["fingers"][finger_index] = segments
        constraints[side] = side_constraints
    return objects, constraints


def update_finger_targets(
    armature,
    objects: dict,
    constraints: dict,
    directions: np.ndarray,
    valid: np.ndarray | None,
    influences: np.ndarray | None,
    frame: int,
) -> None:
    world_scale = float(np.mean(np.abs(armature.matrix_world.to_scale())))
    if world_scale < 1e-8:
        world_scale = 1.0

    for side_index, side in enumerate(("Left", "Right")):
        details = constraints.get(side)
        if details is None:
            continue
        hand_pose = armature.pose.bones[details["hand_name"]]
        hand_rest = armature.data.bones[details["hand_name"]]
        hand_world = armature.matrix_world @ hand_pose.matrix
        hand_rest_inverse = hand_rest.matrix_local.inverted()
        axes_local = details["axes_local"]

        for finger_index, segments in details["fingers"].items():
            first_bone = armature.data.bones[segments[0]["bone_name"]]
            base_local = hand_rest_inverse @ first_bone.head_local
            joint = hand_world @ base_local
            for segment_index, segment in enumerate(segments):
                is_valid = bool(valid[side_index, finger_index, segment_index]) if valid is not None else np.isfinite(directions[side_index, finger_index, segment_index]).all()
                constraint = segment["constraint"]
                influence = float(influences[side_index, finger_index, segment_index]) if influences is not None and is_valid else float(is_valid)
                constraint.influence = influence
                constraint.keyframe_insert(data_path="influence", frame=frame)
                if not is_valid or influence <= 0.0:
                    continue
                source_direction = Vector(directions[side_index, finger_index, segment_index])
                local_direction = sum((axes_local[axis] * source_direction[axis] for axis in range(3)), Vector((0.0, 0.0, 0.0)))
                if local_direction.length < 1e-8:
                    constraint.influence = 0.0
                    constraint.keyframe_insert(data_path="influence", frame=frame)
                    continue
                world_direction = hand_world.to_3x3() @ local_direction.normalized()
                if world_direction.length < 1e-8:
                    constraint.influence = 0.0
                    constraint.keyframe_insert(data_path="influence", frame=frame)
                    continue
                bone = armature.data.bones[segment["bone_name"]]
                joint = joint + world_direction.normalized() * bone.length * world_scale
                target = segment["target"]
                target.location = joint
                target.keyframe_insert(data_path="location", frame=frame)


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


if __name__ == "__main__":
    raise SystemExit(main())
