# Input videos

Place your source videos in this `input` folder on your computer.
Video files are ignored by Git to keep the repository small.

From PowerShell in the project root (`D:\video2glb` on the original computer),
run the following using the project's existing Python environment:

```powershell
.\.venv\Scripts\python.exe convert.py --video ".\input\Passenger.mp4"
```

Replace `Passenger.mp4` with your video's filename.

To process all supported videos in this folder:

```powershell
.\.venv\Scripts\python.exe convert.py --input-dir ".\input" --batch
```

Successful results are written under `output/`; inspect the reported status
for videos that fail validation. See the [project README](../README.md) for
output details and [settings](../config/settings.yaml) for avatar, model,
and Blender paths. Dependencies and Blender must be installed locally.

GitHub displays these instructions; this repository currently has no web
upload form or GitHub workflow for running the converter.
