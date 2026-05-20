from __future__ import annotations

from accent_coach.models import PhonemeInstance
from accent_coach.pipeline.vot import extract_vot


def _make_stop(start: float) -> PhonemeInstance:
    return PhonemeInstance(
        phoneme="t",
        arpabet="T",
        start_time=start,
        end_time=start + 0.05,
        sentence_id=1,
        word="top",
        is_stressed=True,
    )


def test_vot_within_tolerance(synthetic_burst_voice_audio):
    audio, sr = synthetic_burst_voice_audio
    # Burst at 0.02s, voicing at 0.09s → VOT ~70 ms
    stop = _make_stop(0.02)
    vot = extract_vot(audio, sr, stop)
    assert vot is not None, "VOT should be detected"
    # Allow ±20 ms on a synthetic signal (autocorr is approximate)
    assert abs(vot - 70) < 20, f"VOT={vot:.1f} ms too far from 70 ms"
