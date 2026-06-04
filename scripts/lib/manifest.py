"""Manifest loading and path resolution helpers shared across bench/compare scripts."""
from __future__ import annotations

import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    """Load a JSON manifest file and return its contents as a list of dicts."""
    with path.open() as f:
        return json.load(f)


def resolve_path(entry: dict, manifest_path: Path, repo_root: Path) -> Path | None:
    """Resolve a WAV path from a manifest entry.

    Tries three candidates in order:
      1. Raw value as an absolute or CWD-relative path
      2. repo_root / raw value
      3. manifest_path.parent / filename only (same-dir fallback)

    Returns the first existing Path, or None if none found.
    """
    raw = entry.get("wav_path") or entry.get("path", "")
    for candidate in [
        Path(raw),
        repo_root / raw,
        manifest_path.parent / Path(raw).name,
    ]:
        if candidate.exists():
            return candidate
    return None
