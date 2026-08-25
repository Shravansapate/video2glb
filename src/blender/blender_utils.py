from __future__ import annotations

import bpy


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def purge_orphans() -> None:
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def import_avatar(path: str):
    bpy.ops.import_scene.fbx(filepath=path)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in avatar FBX.")
    return armatures[0]


def isolate_target_animation(armature, action) -> None:
    for obj in bpy.context.scene.objects:
        if obj.animation_data and obj != armature:
            obj.animation_data_clear()
    for existing in list(bpy.data.actions):
        if existing != action:
            bpy.data.actions.remove(existing)
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action


def validate_animation_state(armature, required_bones: list[str]) -> None:
    if armature.type != "ARMATURE":
        raise RuntimeError("Animation target is not an armature.")
    missing = [name for name in required_bones if name not in armature.pose.bones]
    if missing:
        raise RuntimeError(f"Required avatar bones missing: {missing}")
    actions = list(bpy.data.actions)
    if len(actions) != 1:
        raise RuntimeError(f"Expected exactly one action, found {len(actions)}.")
    action = actions[0]
    if not action.fcurves:
        raise RuntimeError("Action has no animation channels.")


def export_avatar_glb(path: str, armature) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_frame_range=True,
        export_force_sampling=True,
        export_nla_strips=False,
        export_lights=False,
        export_cameras=False,
    )
