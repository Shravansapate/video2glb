# ISL Video to GLB Converter

Local Windows pipeline for converting one Indian Sign Language source video into an animated GLB using `assets/character.fbx`. The direct path uses MediaPipe in normal Python and runs Blender headlessly only for avatar import, retarget baking, GLB export, and re-import validation. It does not require BVH, Rokoko, or manual Blender steps.

## Run

```powershell
python convert.py --video ".\input\Passenger.mp4"
```

Optional explicit avatar:

```powershell
python convert.py --video ".\input\Passenger.mp4" --avatar ".\assets\character.fbx"
```

Paths are configured in `config/settings.yaml`; no video-specific source edits are required.

## Output

For `Passenger.mp4`, the converter writes:

```text
output/PASSENGER/Passenger.glb
output/PASSENGER/Passenger.pose.npz
output/PASSENGER/Passenger.motion.npz
output/PASSENGER/Passenger.qc.json
output/PASSENGER/Passenger.glb_validation.json
output/PASSENGER/Passenger.source_avatar_validation.json
output/PASSENGER/Passenger.signer_review.json
output/PASSENGER/Passenger.ik_calibration.json
output/PASSENGER/debug/Passenger_pose_overlay.mp4
output/PASSENGER/debug/Passenger_avatar_preview.mp4
output/PASSENGER/debug/Passenger_source_avatar_comparison.mp4
```

The debug videos are controlled by `output.save_debug` or `--save-debug`.

## Checks

The pipeline preserves source FPS, keeps left/right hands separate, saves raw pose data before smoothing, constrains short tracking gaps, uses fixed avatar bone lengths, and validates the exported GLB in a fresh Blender process. The GLB validation checks mesh and armature presence, exactly one animation, timing, wrist-target error, hand visibility, local finger continuity, and local source-to-avatar finger direction error.

`technical_qc: PASS` means the file and tracked motion passed these engineering checks. It does not prove that the animation communicates correct ISL. Every output retains `isl_verified: false` and `production_status: PENDING_SIGNER_REVIEW` until a qualified ISL signer reviews the source and avatar comparison video. The avatar preview uses a high-contrast diagnostic material only; it does not alter the exported GLB.

## Batch

```powershell
python convert.py --input-dir ".\input" --batch
```

Each video is isolated so one failure does not terminate the batch. `output/batch_summary.json` separates technically valid files from signer-approved files. Use batch output for technical processing only until the target vocabulary has been signer-reviewed.

## Legacy Baseline

The direct path never calls Rokoko or BVH. If the previous TDPT/BVH/Rokoko script becomes available, place an unchanged copy at `legacy/bvh_to_glb_rokoko.py` for comparison and fallback research. Current legacy-comparison status is recorded in each output directory rather than fabricated.
