"""Phase G: emit phase0_7_manifest.csv for accent_coach_bench.py.

Experiments:
  A  — synth BC (bc_cal_50) vs RP norms        (50 rows)
  B  — real BC reference clips vs RP norms      (4 rows, one per ref clip)
  C  — owner (users/owner) vs RP norms          (50 rows)
  D  — owner (users/owner) vs synth BC target   (50 rows, same audio, target set)
"""
from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

CAL_50_CSV = pathlib.Path("tts_output/accent_coach/cal_50.csv")
BC_CAL_DIR = pathlib.Path("tts_output/accent_coach/bc_cal_50")
OWNER_DIR = pathlib.Path("tts_output/accent_coach/users/owner")
OUT_CSV = pathlib.Path("tts_output/accent_coach/bench/phase0_7_manifest.csv")

# Real BC reference clips + known transcripts
REAL_BC_REFS = [
    {
        "path": pathlib.Path("tts_output/ref_interview.wav"),
        "transcript": None,  # will be transcribed
        "label": "ref_interview",
    },
    {
        "path": pathlib.Path("tts_output/ref_narrator.wav"),
        "transcript": None,
        "label": "ref_narrator",
    },
    {
        "path": pathlib.Path("tts_output/ref_sherlock.wav"),
        "transcript": None,
        "label": "ref_sherlock",
    },
    {
        "path": pathlib.Path("tts_output/ref_combined.wav"),
        "transcript": None,
        "label": "ref_combined",
    },
]


def transcribe_wav(wav_path: pathlib.Path) -> str:
    """Run whisper via the project venv to get transcript."""
    result = subprocess.run(
        [
            ".venv/bin/python", "-c",
            f"import whisper; m = whisper.load_model('base'); "
            f"r = m.transcribe('{wav_path}', language='en'); print(r['text'].strip())",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def load_cal50() -> list[dict]:
    with CAL_50_CSV.open() as f:
        return list(csv.DictReader(f))


def find_bc_wav(slug: str) -> pathlib.Path | None:
    # bc_cal_50 files named indextts_cal_001.wav, indextts_cal_002.wav, ...
    num = slug.split("_")[1]  # e.g. "001"
    p = BC_CAL_DIR / f"indextts_cal_{num}.wav"
    return p if p.exists() else None


def find_owner_wav(slug: str) -> pathlib.Path | None:
    # owner files named 001_please_leave_..._t.wav etc. — match by numeric prefix
    num = slug.split("_")[1]
    for f in sorted(OWNER_DIR.glob("*.wav")):
        if f.name.startswith(f"{num}_"):
            return f
    return None


def main() -> None:
    cal50 = load_cal50()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing = 0

    # --- Experiment A: synth BC vs RP norms ---
    for i, cal in enumerate(cal50, start=1):
        slug = cal["slug"]
        bc_wav = find_bc_wav(slug)
        if bc_wav is None:
            print(f"  [A] MISSING bc wav for {slug}", file=sys.stderr)
            missing += 1
            continue
        rows.append({
            "experiment": "A",
            "speaker": "synth_bc",
            "label": slug,
            "user_wav": str(bc_wav),
            "target_wav": "",
            "transcript": cal["prompt"],
            "sentence_id": i,
        })

    # --- Experiment B: real BC ref clips vs RP norms ---
    for ref in REAL_BC_REFS:
        if not ref["path"].exists():
            print(f"  [B] MISSING {ref['path']}", file=sys.stderr)
            missing += 1
            continue
        transcript = ref["transcript"]
        if transcript is None:
            print(f"  [B] Transcribing {ref['path']}...", flush=True)
            try:
                transcript = transcribe_wav(ref["path"])
            except Exception as e:  # noqa: BLE001
                print(f"  [B] transcription failed for {ref['path']}: {e}", file=sys.stderr)
                transcript = ""
        rows.append({
            "experiment": "B",
            "speaker": "real_bc",
            "label": ref["label"],
            "user_wav": str(ref["path"]),
            "target_wav": "",
            "transcript": transcript,
            "sentence_id": 0,
        })

    # --- Experiment C: owner vs RP norms ---
    # --- Experiment D: owner vs synth BC target ---
    for i, cal in enumerate(cal50, start=1):
        slug = cal["slug"]
        owner_wav = find_owner_wav(slug)
        bc_wav = find_bc_wav(slug)
        if owner_wav is None:
            print(f"  [C/D] MISSING owner wav for {slug}", file=sys.stderr)
            missing += 1
            continue
        rows.append({
            "experiment": "C",
            "speaker": "owner",
            "label": slug,
            "user_wav": str(owner_wav),
            "target_wav": "",
            "transcript": cal["prompt"],
            "sentence_id": i,
        })
        rows.append({
            "experiment": "D",
            "speaker": "owner",
            "label": slug,
            "user_wav": str(owner_wav),
            "target_wav": str(bc_wav) if bc_wav else "",
            "transcript": cal["prompt"],
            "sentence_id": i,
        })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "speaker", "label", "user_wav", "target_wav",
                        "transcript", "sentence_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} ({len(rows)} rows, {missing} missing)")


if __name__ == "__main__":
    main()
