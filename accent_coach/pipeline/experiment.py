"""Backward-compatible re-exports for accent_coach.pipeline.experiment.

All logic has moved to focused modules:
  generate_clips        → accent_coach.pipeline.generate
  extract_formants      → accent_coach.pipeline.formants
  build_centroids_from_formants,
  load_baseline_centroids,
  score_against         → accent_coach.pipeline.centroids
  score_posthoc_clips   → scripts.lib.scoring.score_clips
"""
from __future__ import annotations

import sys
from pathlib import Path

from accent_coach.pipeline.centroids import (  # noqa: F401
    BASELINE_CENTROIDS_PATH,
    CLEANED_CENTROIDS_PATH,
    build_centroids_from_formants,
    load_baseline_centroids,
    score_against,
)
from accent_coach.pipeline.formants import extract_formants  # noqa: F401
from accent_coach.pipeline.generate import (  # noqa: F401
    INDEXTTS_PYTHON,
    generate_clips,
)

# Ensure scripts/ is on the path so lib.scoring is importable
_SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def score_posthoc_clips(manifest, ecapa_ref, out_csv, n_workers=None, cache=True):
    """Delegate to scripts.lib.scoring.score_clips (single source of truth)."""
    from lib.scoring import score_clips
    return score_clips(manifest, ecapa_ref, out_csv, cache=cache)
