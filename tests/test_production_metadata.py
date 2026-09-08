import json
from pathlib import Path

from convert import write_reports
from src.metadata.production_metadata import (
    atomic_write_json,
    build_production_metadata,
    load_catalog_row,
    merge_metadata,
    normalize_motion_identity,
    sha256_file,
)


def test_motion_identity_normalizes_dataset_variants_without_inventing_domain():
    assert normalize_motion_identity("DELAY")["motion_code"] == "ISL_DELAY_01"
    assert normalize_motion_identity("DELAY (1)")["motion_code"] == "ISL_DELAY_02"
    assert normalize_motion_identity("BEFORE_(SIGN_2)")["motion_code"] == "ISL_BEFORE_02"
    train = normalize_motion_identity("ARRIVE (TRAIN)")
    assert train["motion_code"] == "ISL_ARRIVE_01"
    assert train["context_marker"] == "TRAIN"


def test_merge_preserves_human_values_when_generated_value_is_unknown():
    existing = {"linguistic": {"meaning": "human supplied", "custom": 7}}
    generated = {"linguistic": {"meaning": None, "context": None}}
    assert merge_metadata(existing, generated) == {
        "linguistic": {"meaning": "human supplied", "custom": 7, "context": None}
    }


def test_staged_metadata_merge_refreshes_legacy_and_schema_v3_statuses():
    existing = {
        "technical_qc": "REVIEW",
        "glb_validation": {"status": "REVIEW", "old": True},
        "custom_human_field": "keep me",
    }
    fresh_summary = {
        "technical_qc": "PASS",
        "glb_validation": {"status": "PASS", "file_size": 123},
    }
    schema_v3 = {
        "metadata_schema_version": "3.0",
        "technical_validation": {"technical_qc": "PASS"},
    }

    merged = merge_metadata(merge_metadata(existing, fresh_summary), schema_v3)

    assert merged["technical_qc"] == "PASS"
    assert merged["glb_validation"]["status"] == "PASS"
    assert merged["technical_validation"]["technical_qc"] == "PASS"
    assert merged["custom_human_field"] == "keep me"


def test_atomic_json_write_leaves_no_temporary_file(tmp_path: Path):
    target = tmp_path / "record.json"
    atomic_write_json(target, {"production_eligible": False})
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert list(tmp_path.glob("*.tmp")) == []


def test_catalog_loader_normalizes_declared_boolean_columns(tmp_path: Path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "gloss,provenance_verified,two_handed\nCHANGE,true,false\n",
        encoding="utf-8",
    )

    row = load_catalog_row(catalog_path, "Change")

    assert row["provenance_verified"] is True
    assert row["two_handed"] is False


def test_write_reports_compacts_constant_rate_timestamps(tmp_path: Path):
    class VideoInfo:
        def to_json_dict(self):
            return {
                "fps": 25.0,
                "frame_count": 3,
                "timestamps_ms": [0, 40, 80],
            }

    class QCResult:
        technical_qc = "PASS"

        def to_json_dict(self):
            return {"technical_qc": self.technical_qc, "metrics": {}}

    metadata_path = tmp_path / "sample.metadata.json"
    write_reports(metadata_path, tmp_path / "sample.qc.json", VideoInfo(), QCResult())
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "timestamps_ms" not in payload["video"]
    assert payload["video"]["timestamp_start_ms"] == 0
    assert payload["video"]["timestamp_end_ms"] == 80
    assert payload["video"]["timestamp_interval_ms"] == 40.0


def test_build_metadata_derives_approved_production_state_from_release_gate(tmp_path: Path):
    source = _write_asset(tmp_path / "Change.mp4", b"source")
    avatar = _write_asset(tmp_path / "character.fbx", b"avatar")
    glb = _write_asset(tmp_path / "Change.glb", b"glb")
    comparison = _write_asset(tmp_path / "comparison.mp4", b"comparison")
    source_hash = sha256_file(source)
    glb_hash = sha256_file(glb)
    comparison_hash = sha256_file(comparison)

    glb_validation = {
        "status": "PASS",
        "validation_run_id": "glb-validation-run-001",
        "validated_glb_sha256": glb_hash,
        "khronos_validation": {
            "status": "PASS",
            "validated_glb_sha256": glb_hash,
            "validator_version": "2.0.0-dev.3.10",
            "counts": {"numErrors": 0},
        },
        "animation_count": 1,
        "full_clip_collision": {
            "status": "PASS",
            "evaluated_every_input_frame": True,
            "frames": {"total": 3, "evaluated": 3, "evaluated_coverage": 1.0},
        },
    }
    source_avatar_validation = {
        "status": "PASS",
        "validation_run_id": "source-avatar-validation-run-001",
        "source_video_sha256": source_hash,
        "comparison_video_sha256": comparison_hash,
        "source_frame_count": 3,
    }
    signer_review = {
        "signer_verdict": "PASS",
        "isl_verified": True,
        "reviewer": "Qualified reviewer",
        "reviewed_at": "2026-09-03T09:00:00Z",
        "hash_binding": {
            "source_video_sha256": source_hash,
            "glb_sha256": glb_hash,
            "comparison_video_sha256": comparison_hash,
        },
        "collision_review": {"status": "PASS"},
        "signature_verification": {"status": "PASS", "verified": True},
    }
    catalog = {
        "meaning": "change",
        "license": "licensed",
        "source_reference": "catalog://change/1",
        "provenance_verified": True,
    }

    metadata = _build_metadata(
        tmp_path,
        source=source,
        avatar=avatar,
        glb=glb,
        comparison=comparison,
        glb_validation=glb_validation,
        source_avatar_validation=source_avatar_validation,
        signer_review=signer_review,
        catalog=catalog,
    )

    production = metadata["production"]
    assert production["production_status"] == "APPROVED"
    assert production["production_eligible"] is True
    assert production["engineering_candidate"] is True
    assert production["release_gate"]["production_eligible"] is True
    assert production["release_gate"]["blockers"] == []
    assert metadata["file_integrity"]["comparison_video_sha256"] == comparison_hash

    calibration_review = _build_metadata(
        tmp_path,
        source=source,
        avatar=avatar,
        glb=glb,
        comparison=comparison,
        glb_validation=glb_validation,
        source_avatar_validation=source_avatar_validation,
        signer_review=signer_review,
        catalog=catalog,
        neutral_hand_validation={"status": "REVIEW"},
    )
    calibration_gate = calibration_review["production"]["release_gate"]
    assert calibration_gate["checks"]["neutral_hand_calibration_pass"] is False
    assert calibration_review["production"]["engineering_candidate"] is False


def test_build_metadata_cannot_be_approved_by_truthy_boolean_strings(tmp_path: Path):
    source = _write_asset(tmp_path / "Change.mp4", b"source")
    avatar = _write_asset(tmp_path / "character.fbx", b"avatar")
    glb = _write_asset(tmp_path / "Change.glb", b"glb")
    comparison = _write_asset(tmp_path / "comparison.mp4", b"comparison")

    glb_validation = {
        "status": "PASS",
        "full_clip_collision": {
            "status": "PASS",
            "evaluated_every_input_frame": True,
            "frames": {"total": 3, "evaluated": 3, "evaluated_coverage": 1.0},
        },
    }
    signer_review = {
        "signer_verdict": "PASS",
        "isl_verified": "true",
        "reviewer": "Reviewer",
        "reviewed_at": "2026-09-03T09:00:00Z",
        "hash_binding": {
            "source_video_sha256": sha256_file(source),
            "glb_sha256": sha256_file(glb),
            "comparison_video_sha256": sha256_file(comparison),
        },
        "collision_review": {"status": "PASS"},
    }
    catalog = {
        "meaning": "change",
        "license": "licensed",
        "source_reference": "catalog://change/1",
        "provenance_verified": "true",
    }

    metadata = _build_metadata(
        tmp_path,
        source=source,
        avatar=avatar,
        glb=glb,
        comparison=comparison,
        glb_validation=glb_validation,
        source_avatar_validation={"status": "PASS", "source_frame_count": 3},
        signer_review=signer_review,
        catalog=catalog,
    )

    release_gate = metadata["production"]["release_gate"]
    assert metadata["isl_validation"]["isl_verified"] is False
    assert metadata["source"]["provenance_verified"] is False
    assert release_gate["checks"]["isl_verified"] is False
    assert release_gate["checks"]["provenance_verified"] is False
    assert metadata["production"]["production_eligible"] is False


def test_logical_identity_survives_role_named_source_snapshot(tmp_path: Path):
    source = _write_asset(tmp_path / "source_video.mp4", b"source")
    avatar = _write_asset(tmp_path / "character.fbx", b"avatar")
    glb = _write_asset(tmp_path / "Change.glb", b"glb")
    comparison = _write_asset(tmp_path / "comparison.mp4", b"comparison")

    metadata = _build_metadata(
        tmp_path,
        source=source,
        avatar=avatar,
        glb=glb,
        comparison=comparison,
        glb_validation={},
        source_avatar_validation={},
        signer_review={},
        catalog={},
        logical_identity_stem="Change",
    )

    assert metadata["motion_identity"]["gloss"] == "CHANGE"
    assert metadata["motion_identity"]["motion_code"] == "ISL_CHANGE_01"
    assert metadata["animation"]["animation_name"] == "CHANGE"


def test_metadata_records_real_execution_timing_and_unknown_review_fields(tmp_path: Path):
    source = _write_asset(tmp_path / "Arrive.mp4", b"source")
    avatar = _write_asset(tmp_path / "character.fbx", b"avatar")
    glb = _write_asset(tmp_path / "Arrive.glb", b"glb")
    comparison = _write_asset(tmp_path / "comparison.mp4", b"comparison")
    execution = {
        "run_id": "a" * 32, "batch_id": "b" * 32, "attempt": 2,
        "original_filename": "Arrive.mp4", "context_fingerprint": "c" * 64,
        "stage_cache": [{"stage": "tracking", "reused": True}],
        "stages": [{"name": "Tracking", "status": "COMPLETED", "elapsed_seconds": 1.2}],
        "calibration_hashes": {"profile": "d" * 64},
    }
    preparation = {"normalized": True, "source": {"frame_count": 4}, "working": {"frame_count": 5}}
    metadata = _build_metadata(tmp_path, source=source, avatar=avatar, glb=glb,
        comparison=comparison, glb_validation={"root_drift": {"status": "PASS"},
            "neutral_finger_shape": {"status": "PASS", "evaluated_frames": 3}},
        source_avatar_validation={}, signer_review={}, catalog={},
        execution_context=execution, preparation=preparation)
    assert metadata["metadata_schema_version"] == "3.1"
    assert metadata["asset_identity"]["video_id"] == sha256_file(source)
    assert metadata["asset_identity"]["batch_id"] == "b" * 32
    assert metadata["processing"]["elapsed_seconds"] == 60
    assert metadata["processing"]["attempt"] == 2
    assert metadata["processing"]["stage_cache"][0]["reused"] is True
    assert metadata["preparation"]["source"]["frame_count"] == 4
    assert metadata["preparation"]["working"]["frame_count"] == 5
    assert metadata["technical_validation"]["motion_quality"]["root_drift"]["status"] == "PASS"
    assert metadata["technical_validation"]["motion_quality"]["neutral_finger_shape"] == {
        "status": "PASS", "evaluated_frames": 3,
    }
    assert metadata["source"]["license"] is None
    assert metadata["linguistic"]["meaning"] is None
    assert metadata["isl_validation"]["reviewer"] is None
    assert metadata["animation"]["has_finger_motion"] is False


def _write_asset(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _build_metadata(
    root: Path,
    *,
    source: Path,
    avatar: Path,
    glb: Path,
    comparison: Path,
    glb_validation: dict,
    source_avatar_validation: dict,
    signer_review: dict,
    catalog: dict,
    neutral_hand_validation: dict | None = None,
    logical_identity_stem: str | None = None,
    execution_context: dict | None = None,
    preparation: dict | None = None,
) -> dict:
    return build_production_metadata(
        project_root=root,
        source_video=source,
        avatar_file=avatar,
        glb_path=glb,
        pose_path=root / "Change.pose.npz",
        motion_path=root / "Change.motion.npz",
        qc_path=root / "Change.qc.json",
        glb_validation_path=root / "Change.glb_validation.json",
        source_avatar_validation_path=root / "Change.source_avatar_validation.json",
        signer_review_path=root / "Change.signer_review.json",
        comparison_video_path=comparison,
        video_info={"fps": 25.0, "frame_count": 3, "timestamps_ms": [0, 40, 80]},
        tracking_qc={"technical_qc": "PASS", "metrics": {}},
        glb_validation=glb_validation,
        source_avatar_validation=source_avatar_validation,
        signer_review=signer_review,
        avatar_profile={},
        bone_map={},
        neutral_hand_validation=(
            {"status": "PASS"}
            if neutral_hand_validation is None
            else neutral_hand_validation
        ),
        neutral_hand_pose_path=None,
        started_at="2026-09-03T08:00:00Z",
        completed_at="2026-09-03T08:01:00Z",
        catalog_row=catalog,
        logical_identity_stem=logical_identity_stem,
        execution_context=execution_context,
        preparation=preparation,
    )
