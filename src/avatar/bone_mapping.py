from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_bone_map(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = payload.get("required_missing") or []
    if missing:
        raise ValueError(f"Avatar bone map is incomplete: {missing}")
    mapping = payload.get("map")
    if not isinstance(mapping, dict):
        raise ValueError(f"Avatar bone map has no 'map' object: {path}")
    return {str(key): str(value) for key, value in mapping.items()}


def load_avatar_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
