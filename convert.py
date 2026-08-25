from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.avatar.bone_mapping import load_bone_map
from src.motion.skeleton_solver import solve_motion_from_pose
from src.qc.source_avatar_comparison import create_source_avatar_comparison
from src.tracking.holistic_tracker import MediaPipeHolisticBackend
from src.tracking.pose_schema import PoseSequence
from src.tracking.tracking_qc import evaluate_tracking
from src.video.inspector import inspect_video


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    args = parse_args()
    if args.revalidate_failed:
        return revalidate_failed_outputs(args)
    if args.batch:
        return run_batch(args)

    if not args.video:
        raise SystemExit("--video is required unless --batch is used.")

    config = load_yaml(resolve_project_path(args.config))
    qc_thresholds = load_yaml(resolve_project_path(args.qc_thresholds))

    video_path = resolve_project_path(args.video)
    avatar_path = resolve_project_path(args.avatar or config["avatar"]["path"])
    model_path = resolve_project_path(config["pose"]["model_path"])
    gloss = video_path.stem
    output_dir_name = gloss.upper() if config.get("output", {}).get("uppercase_output_dir", True) else gloss
    output_dir = PROJECT_ROOT / "output" / output_dir_name
    debug_dir = output_dir / "debug"
    pose_path = output_dir / f"{gloss}.pose.npz"
    body_motion_path = output_dir / f"{gloss}_body_only.motion.npz"
    body_glb_path = output_dir / f"{gloss}_body_only.glb"
    palm_motion_path = output_dir / f"{gloss}_palms.motion.npz"
    palm_glb_path = output_dir / f"{gloss}_palms.glb"
    motion_path = output_dir / f"{gloss}.motion.npz"
    glb_path = output_dir / f"{gloss}.glb"
    metadata_path = output_dir / f"{gloss}.metadata.json"
    qc_path = output_dir / f"{gloss}.qc.json"
    glb_validation_path = output_dir / f"{gloss}.glb_validation.json"
    ik_report_path = output_dir / f"{gloss}.ik_calibration.json"
    comparison_path = output_dir / f"{gloss}.comparison.json"
    source_avatar_validation_path = output_dir / f"{gloss}.source_avatar_validation.json"
    match_validation_path = output_dir / f"{gloss}.match_validation.json"
    signer_review_path = output_dir / f"{gloss}.signer_review.json"
    overlay_path = debug_dir / f"{gloss}_pose_overlay.mp4"
    avatar_preview_path = debug_dir / f"{gloss}_avatar_preview.mp4"
    source_avatar_preview_path = debug_dir / f"{gloss}_source_avatar_comparison.mp4"
    profile_path = PROJECT_ROOT / "config" / "avatar_profile.json"
    bone_map_path = PROJECT_ROOT / "config" / "avatar_bone_map.json"
    blender_path = resolve_project_path(config.get("blender", {}).get("executable") or default_blender_path())

    print_header(video_path, avatar_path)

    try:
        video_info = run_stage(1, "Inspect video", lambda: inspect_video(video_path))
        ensure_required_inputs(video_path, avatar_path, model_path)

        def process_pose():
            backend = MediaPipeHolisticBackend(model_path=model_path, options=config.get("pose", {}))
            return backend.process_video(
                video_path=video_path,
                video_info=video_info,
                overlay_path=overlay_path if should_save_debug(args, config) else None,
                progress=print_tracking_progress,
            )

        pose_sequence = run_stage(2, "Extract holistic landmarks", process_pose)
        run_stage(3, "Save raw pose", lambda: pose_sequence.save_npz(pose_path))
        qc_result = run_stage(4, "Validate tracking", lambda: evaluate_tracking(pose_sequence, qc_thresholds))
        run_stage(5, "Calibrate avatar", lambda: ensure_avatar_calibrated(blender_path, avatar_path, profile_path, bone_map_path))
        run_stage(
            6,
            "Solve torso/arms",
            lambda: solve_motion_from_pose(
                pose_path,
                profile_path,
                bone_map_path,
                body_motion_path,
                gloss,
                body_only=True,
                include_palms=False,
                include_fingers=False,
                smoothing=False,
            ),
        )
        run_stage(7, "Export body GLB", lambda: run_blender_apply(blender_path, avatar_path, profile_path, bone_map_path, body_motion_path, body_glb_path))
        run_stage(
            8,
            "Solve palms",
            lambda: solve_motion_from_pose(
                pose_path,
                profile_path,
                bone_map_path,
                palm_motion_path,
                gloss,
                body_only=False,
                include_palms=True,
                include_fingers=False,
                smoothing=False,
            ),
        )
        run_stage(9, "Export palm GLB", lambda: run_blender_apply(blender_path, avatar_path, profile_path, bone_map_path, palm_motion_path, palm_glb_path))
        run_stage(
            10,
            "Solve fingers/QC motion",
            lambda: solve_motion_from_pose(
                pose_path,
                profile_path,
                bone_map_path,
                motion_path,
                gloss,
                body_only=False,
                include_palms=True,
                include_fingers=True,
                smoothing=bool(config.get("processing", {}).get("smoothing", True)),
            ),
        )
        run_stage(
            11,
            "Launch Blender/export GLB",
            lambda: run_blender_apply(
                blender_path,
                avatar_path,
                profile_path,
                bone_map_path,
                motion_path,
                glb_path,
                ik_report_path,
            ),
        )
        glb_validation = run_stage(
            12,
            "Re-import GLB",
            lambda: run_blender_validate(
                blender_path,
                glb_path,
                motion_path,
                profile_path,
                bone_map_path,
                glb_validation_path,
            ),
        )
        source_avatar_validation: dict[str, Any] = {
            "status": "NOT_RENDERED",
            "reason": "Debug rendering was disabled; no source/avatar comparison video was requested.",
            "isl_verified": False,
            "isl_verification_status": "PENDING_SIGNER_REVIEW",
        }
        if should_save_debug(args, config):
            run_stage(
                13,
                "Render avatar preview",
                lambda: run_blender_render_animation(blender_path, glb_path, video_info.fps, avatar_preview_path),
            )
            source_avatar_validation = run_stage(
                14,
                "Compare source/avatar",
                lambda: create_source_avatar_comparison(video_path, avatar_preview_path, source_avatar_preview_path),
            )
        final_technical_qc = max_status(qc_result.technical_qc, glb_validation.get("status", "FAIL"))
        if source_avatar_validation.get("status") in {"PASS", "REVIEW", "FAIL"}:
            final_technical_qc = max_status(final_technical_qc, source_avatar_validation["status"])
        signer_review = build_signer_review_record(
            gloss=gloss,
            technical_qc=final_technical_qc,
            source_video=video_path,
            glb_path=glb_path,
            comparison_video=source_avatar_preview_path if should_save_debug(args, config) else None,
        )
        run_stage(
            15,
            "Write comparison/review reports",
            lambda: write_comparison_and_review_reports(
                source_avatar_validation_path,
                match_validation_path,
                signer_review_path,
                source_avatar_validation,
                signer_review,
            ),
        )
        run_stage(
            16,
            "Write metadata/QC",
            lambda: write_reports(
                metadata_path,
                qc_path,
                video_info,
                qc_result,
                glb_validation,
                final_technical_qc,
                source_avatar_validation,
                signer_review,
            ),
        )
        run_stage(17, "Record research status", lambda: write_research_status(comparison_path, gloss, glb_path))
        if glb_validation.get("status") == "FAIL":
            move_failed_output(glb_path)
            raise RuntimeError(f"GLB validation failed: {glb_validation.get('reasons')}")
    except Exception as exc:
        print(f"\nFAIL: {exc}")
        return 1

    print("\nOUTPUT:")
    print(relative_display(glb_path))
    print(relative_display(pose_path))
    print(relative_display(motion_path))
    print(relative_display(qc_path))
    print(relative_display(source_avatar_validation_path))
    print(relative_display(signer_review_path))
    if ik_report_path.exists():
        print(relative_display(ik_report_path))
    if should_save_debug(args, config):
        print(relative_display(overlay_path))
        print(relative_display(avatar_preview_path))
        print(relative_display(source_avatar_preview_path))
    print("\nTechnical QC:")
    print(final_technical_qc)
    print("\nISL Verification:")
    print("PENDING")
    print(f"\nFinal {gloss} integration test attempted.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISL video to GLB converter, built milestone-by-milestone.")
    parser.add_argument("--video", help="Input MP4 path.")
    parser.add_argument("--input-dir", help="Directory of MP4 files for batch mode.")
    parser.add_argument("--batch", action="store_true", help="Process every MP4 in --input-dir without stopping on individual failures.")
    parser.add_argument("--avatar", help="Optional avatar FBX path. Defaults to config/settings.yaml.")
    parser.add_argument("--config", default="./config/settings.yaml", help="Project settings YAML.")
    parser.add_argument("--qc-thresholds", default="./config/qc_thresholds.yaml", help="Tracking QC thresholds YAML.")
    parser.add_argument("--save-debug", action="store_true", help="Force writing debug overlay video.")
    parser.add_argument(
        "--revalidate-failed",
        action="store_true",
        help="Re-check preserved failed GLBs after a validator fix without re-running video tracking.",
    )
    return parser.parse_args()


def run_batch(args: argparse.Namespace) -> int:
    if not args.input_dir:
        raise SystemExit("--input-dir is required with --batch.")
    input_dir = resolve_project_path(args.input_dir)
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 files found in {input_dir}")
    summary = {
        "total": len(videos),
        "pass": 0,
        "review": 0,
        "fail": 0,
        "production_ready": 0,
        "pending_signer_review": 0,
        "items": [],
    }
    for video in videos:
        cmd = ["python", str(PROJECT_ROOT / "convert.py"), "--video", str(video), "--config", args.config, "--qc-thresholds", args.qc_thresholds]
        if args.avatar:
            cmd.extend(["--avatar", args.avatar])
        if args.save_debug:
            cmd.append("--save-debug")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        qc_path = PROJECT_ROOT / "output" / video.stem.upper() / f"{video.stem}.qc.json"
        status = "FAIL"
        isl_verified = False
        if result.returncode == 0 and qc_path.exists():
            try:
                qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
                status = qc_payload.get("technical_qc", "FAIL")
                isl_verified = bool(qc_payload.get("isl_verified", False))
            except json.JSONDecodeError:
                status = "FAIL"
        status_key = status.lower() if status in {"PASS", "REVIEW", "FAIL"} else "fail"
        summary[status_key] += 1
        if status == "PASS" and isl_verified:
            summary["production_ready"] += 1
            production_status = "APPROVED"
        elif status != "FAIL":
            summary["pending_signer_review"] += 1
            production_status = "PENDING_SIGNER_REVIEW"
        else:
            production_status = "NOT_ELIGIBLE"
        summary["items"].append(
            {
                "video": str(video),
                "status": status,
                "isl_verified": isl_verified,
                "production_status": production_status,
                "returncode": result.returncode,
            }
        )
    summary_path = PROJECT_ROOT / "output" / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: summary[key] for key in ["total", "pass", "review", "fail", "production_ready", "pending_signer_review"]},
            indent=2,
        )
    )
    return 0 if summary["fail"] == 0 else 1


def revalidate_failed_outputs(args: argparse.Namespace) -> int:
    """Recover valid/reviewable GLBs after a validation-only defect is fixed."""
    config = load_yaml(resolve_project_path(args.config))
    thresholds = load_yaml(resolve_project_path(args.qc_thresholds))
    blender_path = resolve_project_path(config.get("blender", {}).get("executable") or default_blender_path())
    failed_root = PROJECT_ROOT / "failed"
    input_dir = resolve_project_path(args.input_dir or "./input")
    input_videos = {video.stem.lower(): video for video in input_dir.glob("*.mp4")}
    profile_path = PROJECT_ROOT / "config" / "avatar_profile.json"
    bone_map_path = PROJECT_ROOT / "config" / "avatar_bone_map.json"
    summary = {"total": 0, "pass": 0, "review": 0, "fail": 0, "recovered": 0, "items": []}

    for failed_glb in sorted(failed_root.glob("*/*.glb")):
        summary["total"] += 1
        output_dir = PROJECT_ROOT / "output" / failed_glb.parent.name
        gloss = failed_glb.stem
        motion_path = output_dir / f"{gloss}.motion.npz"
        pose_path = output_dir / f"{gloss}.pose.npz"
        validation_path = output_dir / f"{gloss}.glb_validation.json"
        qc_path = output_dir / f"{gloss}.qc.json"
        metadata_path = output_dir / f"{gloss}.metadata.json"
        source_avatar_path = output_dir / f"{gloss}.source_avatar_validation.json"
        match_path = output_dir / f"{gloss}.match_validation.json"
        signer_review_path = output_dir / f"{gloss}.signer_review.json"
        video_path = input_videos.get(gloss.lower())
        item: dict[str, Any] = {"video": relative_display(failed_glb), "status": "FAIL", "recovered": False}

        try:
            if not motion_path.exists() or not pose_path.exists():
                raise FileNotFoundError("Required saved motion or pose file is missing.")
            validation = validate_existing_glb(
                blender_path,
                failed_glb,
                motion_path,
                profile_path,
                bone_map_path,
                validation_path,
            )
            status = str(validation.get("status", "FAIL"))
            item["status"] = status
            output_glb = failed_glb
            if status != "FAIL":
                recovered_glb = output_dir / failed_glb.name
                if recovered_glb.exists():
                    raise RuntimeError(f"Refusing to overwrite an existing output GLB: {recovered_glb}")
                shutil.move(str(failed_glb), str(recovered_glb))
                output_glb = recovered_glb
                item["recovered"] = True
                summary["recovered"] += 1

            sequence = PoseSequence.load_npz(pose_path)
            tracking_qc = evaluate_tracking(sequence, thresholds)
            source_validation = read_json(source_avatar_path, default_source_avatar_validation())
            final_qc = max_status(tracking_qc.technical_qc, status)
            signer_review = build_signer_review_record(
                gloss=gloss,
                technical_qc=final_qc,
                source_video=video_path or Path(f"input/{gloss}.mp4"),
                glb_path=output_glb,
                comparison_video=(output_dir / "debug" / f"{gloss}_source_avatar_comparison.mp4"),
            )
            write_comparison_and_review_reports(
                source_avatar_path,
                match_path,
                signer_review_path,
                source_validation,
                signer_review,
            )
            if video_path is not None and video_path.exists():
                write_reports(
                    metadata_path,
                    qc_path,
                    inspect_video(video_path),
                    tracking_qc,
                    validation,
                    final_qc,
                    source_validation,
                    signer_review,
                )
        except Exception as exc:
            item["error"] = str(exc)
            status = "FAIL"

        status_key = status.lower() if status in {"PASS", "REVIEW", "FAIL"} else "fail"
        summary[status_key] += 1
        summary["items"].append(item)

    summary_path = PROJECT_ROOT / "output" / "revalidation_summary.json"
    refreshed_batch = summarize_existing_batch(sorted(input_videos.values()))
    write_json(PROJECT_ROOT / "output" / "batch_summary.json", refreshed_batch)
    summary["batch_after_revalidation"] = {
        key: refreshed_batch[key]
        for key in ("total", "pass", "review", "fail", "production_ready", "pending_signer_review")
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["fail"] == 0 else 1


def summarize_existing_batch(videos: list[Path]) -> dict[str, Any]:
    """Build a current batch summary from already processed output folders."""
    summary = {
        "total": len(videos),
        "pass": 0,
        "review": 0,
        "fail": 0,
        "production_ready": 0,
        "pending_signer_review": 0,
        "items": [],
    }
    for video in videos:
        output_dir = PROJECT_ROOT / "output" / video.stem.upper()
        qc_payload = read_json(output_dir / f"{video.stem}.qc.json", {})
        status = str(qc_payload.get("technical_qc", "FAIL"))
        isl_verified = bool(qc_payload.get("isl_verified", False))
        if status not in {"PASS", "REVIEW", "FAIL"}:
            status = "FAIL"
        summary[status.lower()] += 1
        if status == "PASS" and isl_verified:
            production_status = "APPROVED"
            summary["production_ready"] += 1
        elif status != "FAIL":
            production_status = "PENDING_SIGNER_REVIEW"
            summary["pending_signer_review"] += 1
        else:
            production_status = "NOT_ELIGIBLE"
        summary["items"].append(
            {
                "video": str(video),
                "status": status,
                "isl_verified": isl_verified,
                "production_status": production_status,
            }
        )
    return summary


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def ensure_required_inputs(video_path: Path, avatar_path: Path, model_path: Path) -> None:
    missing = [path for path in (video_path, avatar_path, model_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required input missing: {names}")


def run_stage(index: int, label: str, func):
    print(f"[{index}] {label:<29}", end="", flush=True)
    result = func()
    print(" PASS")
    return result


def print_tracking_progress(done: int, total: int) -> None:
    if total:
        print(f"\r[2] Extract holistic landmarks ... {done}/{total}", end="", flush=True)
    else:
        print(f"\r[2] Extract holistic landmarks ... {done}", end="", flush=True)


def ensure_avatar_calibrated(blender_path: Path, avatar_path: Path, profile_path: Path, bone_map_path: Path) -> None:
    needs_calibration = True
    if profile_path.exists() and bone_map_path.exists():
        try:
            load_bone_map(bone_map_path)
            needs_calibration = False
        except Exception:
            needs_calibration = True
    if not needs_calibration:
        return
    cmd = [
        "python",
        str(PROJECT_ROOT / "calibrate_avatar.py"),
        "--blender",
        str(blender_path),
        "--avatar",
        str(avatar_path),
        "--profile",
        str(profile_path),
        "--bone-map",
        str(bone_map_path),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    load_bone_map(bone_map_path)


def run_blender_apply(
    blender_path: Path,
    avatar_path: Path,
    profile_path: Path,
    bone_map_path: Path,
    motion_path: Path,
    output_path: Path,
    ik_report_path: Path | None = None,
) -> None:
    ensure_blender(blender_path)
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(PROJECT_ROOT / "src" / "blender" / "blender_apply_motion.py"),
        "--",
        "--avatar",
        str(avatar_path),
        "--profile",
        str(profile_path),
        "--bone-map",
        str(bone_map_path),
        "--motion",
        str(motion_path),
        "--output",
        str(output_path),
    ]
    # The direct solver stores body-relative wrist targets. Blender IK uses
    # the avatar's fixed bone lengths to reach those targets without stretching.
    cmd.append("--use-ik")
    # Hand-world landmarks are stored only as local finger directions. Blender
    # attaches those directions to the solved avatar palm before baking.
    cmd.append("--use-finger-tracking")
    if ik_report_path is not None:
        cmd.extend(["--ik-report", str(ik_report_path)])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Blender did not create a non-empty GLB: {output_path}")


def run_blender_validate(
    blender_path: Path,
    glb_path: Path,
    motion_path: Path,
    profile_path: Path,
    bone_map_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    ensure_blender(blender_path)
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(PROJECT_ROOT / "src" / "blender" / "blender_validate_glb.py"),
        "--",
        "--glb",
        str(glb_path),
        "--motion",
        str(motion_path),
        "--profile",
        str(profile_path),
        "--bone-map",
        str(bone_map_path),
        "--report",
        str(report_path),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def validate_existing_glb(
    blender_path: Path,
    glb_path: Path,
    motion_path: Path,
    profile_path: Path,
    bone_map_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Return Blender's report even when validation deliberately exits non-zero."""
    try:
        return run_blender_validate(blender_path, glb_path, motion_path, profile_path, bone_map_path, report_path)
    except subprocess.CalledProcessError:
        if report_path.exists():
            return read_json(report_path, {"status": "FAIL", "reasons": ["GLB validation process failed."]})
        raise


def run_blender_render_animation(blender_path: Path, glb_path: Path, fps: float, output_path: Path) -> None:
    ensure_blender(blender_path)
    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(PROJECT_ROOT / "src" / "blender" / "blender_render_animation.py"),
        "--",
        "--glb",
        str(glb_path),
        "--output",
        str(output_path),
        "--fps",
        str(fps),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Blender did not create a non-empty avatar preview: {output_path}")


def ensure_blender(blender_path: Path) -> None:
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_path}")


def move_failed_output(glb_path: Path) -> None:
    if not glb_path.exists():
        return
    failed_dir = PROJECT_ROOT / "failed" / glb_path.parent.name
    failed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(glb_path), str(failed_dir / glb_path.name))


def write_reports(
    metadata_path: Path,
    qc_path: Path,
    video_info,
    qc_result,
    glb_validation: dict[str, Any] | None = None,
    final_technical_qc: str | None = None,
    source_avatar_validation: dict[str, Any] | None = None,
    signer_review: dict[str, Any] | None = None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    final_technical_qc = final_technical_qc or qc_result.technical_qc
    metadata = {
        "video": video_info.to_json_dict(),
        "milestone": "Passenger direct MP4 to GLB integration",
        "pipeline_stage": "video_to_pose_to_motion_to_glb",
        "technical_qc": final_technical_qc,
        "isl_verified": False,
        "production_status": (signer_review or {}).get("production_status", "PENDING_SIGNER_REVIEW"),
        "glb_validation": glb_validation or {},
        "source_avatar_validation": source_avatar_validation or {},
        "signer_review": signer_review or {},
    }
    qc_payload = qc_result.to_json_dict()
    qc_payload["technical_qc"] = final_technical_qc
    qc_payload["glb_validation"] = glb_validation or {}
    qc_payload["source_avatar_validation"] = source_avatar_validation or {}
    qc_payload["production_status"] = (signer_review or {}).get("production_status", "PENDING_SIGNER_REVIEW")
    if glb_validation and glb_validation.get("status") == "REVIEW":
        qc_payload["review_reasons"] = list(qc_payload.get("review_reasons", [])) + list(glb_validation.get("review_reasons", []))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def default_source_avatar_validation() -> dict[str, Any]:
    return {
        "status": "NOT_RENDERED",
        "reason": "No source/avatar comparison report was available during validation recovery.",
        "isl_verified": False,
        "isl_verification_status": "PENDING_SIGNER_REVIEW",
    }


def build_signer_review_record(
    gloss: str,
    technical_qc: str,
    source_video: Path,
    glb_path: Path,
    comparison_video: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gloss": gloss.upper(),
        "technical_qc": technical_qc,
        "isl_verified": False,
        "production_status": "PENDING_SIGNER_REVIEW" if technical_qc != "FAIL" else "NOT_ELIGIBLE",
        "source_video": relative_display(source_video),
        "avatar_glb": relative_display(glb_path),
        "comparison_video": relative_display(comparison_video) if comparison_video else None,
        "signer_verdict": "PENDING",
        "reviewer": None,
        "reviewed_at": None,
        "notes": None,
        "approval_rule": "A qualified ISL signer must mark this sign PASS before it is released for passenger-facing use.",
    }


def write_comparison_and_review_reports(
    source_avatar_validation_path: Path,
    match_validation_path: Path,
    signer_review_path: Path,
    source_avatar_validation: dict[str, Any],
    signer_review: dict[str, Any],
) -> None:
    """Write current technical timing and human-review records together."""
    current_match = {
        **source_avatar_validation,
        "report_kind": "CURRENT_DIRECT_PIPELINE_TIMING_COMPATIBILITY",
        "summary": "This supersedes stale render-comparison reports. It verifies timing and frame alignment, not ISL meaning.",
    }
    write_json(source_avatar_validation_path, source_avatar_validation)
    write_json(match_validation_path, current_match)
    write_json(signer_review_path, signer_review)


def write_research_status(path: Path, gloss: str, new_glb_path: Path) -> None:
    legacy_script = PROJECT_ROOT / "legacy" / "bvh_to_glb_rokoko.py"
    payload = {
        "gloss": gloss.upper(),
        "new_pipeline": {
            "status": "PASS" if new_glb_path.exists() else "FAIL",
            "glb": relative_display(new_glb_path),
        },
        "old_pipeline": {
            "status": "UNAVAILABLE",
            "reason": "legacy/bvh_to_glb_rokoko.py was not present; TDPT/BVH/Rokoko comparison was not run.",
        },
        "comparison_metrics": {
            "arm_trajectory": "PENDING_LEGACY_BASELINE",
            "palm_orientation": "PENDING_LEGACY_BASELINE",
            "finger_accuracy": "PENDING_LEGACY_BASELINE",
            "jitter": "PENDING_LEGACY_BASELINE",
            "timing": "PENDING_LEGACY_BASELINE",
            "processing_effort": "PENDING_LEGACY_BASELINE",
        },
    }
    if legacy_script.exists():
        payload["old_pipeline"]["status"] = "PRESENT_NOT_RUN"
        payload["old_pipeline"]["reason"] = "Legacy script exists, but automated TDPT/BVH source generation is not configured."
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def should_save_debug(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    return bool(args.save_debug or config.get("output", {}).get("save_debug", True))


def print_header(video_path: Path, avatar_path: Path) -> None:
    print("=" * 51)
    print("ISL VIDEO -> GLB CONVERTER")
    print("=" * 51)
    print("\nInput:")
    print(video_path.name)
    print("\nAvatar:")
    print(avatar_path.name)
    print()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def default_blender_path() -> str:
    return "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"


def relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def max_status(current: str, candidate: str) -> str:
    order = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
    return candidate if order.get(candidate, 2) > order.get(current, 2) else current


if __name__ == "__main__":
    raise SystemExit(main())
