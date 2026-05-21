"""Phase E: BBC speaker gender audit via F0 distribution.

Outputs:
  tts_output/modern_rp_corpus/bbc/speaker_audit.csv
  docs/img/accent_coach_phase0_7_bbc_f0_hist.png
Prints: n_male / n_female / n_ambiguous counts + decision.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import parselmouth

CLIPS_DIR = pathlib.Path("tts_output/modern_rp_corpus/bbc/clips")
OUT_CSV = pathlib.Path("tts_output/modern_rp_corpus/bbc/speaker_audit.csv")
OUT_PLOT = pathlib.Path("docs/img/accent_coach_phase0_7_bbc_f0_hist.png")

FEMALE_F0_THRESH = 175.0  # Hz — above this → female if voiced_fraction > 0.3
MALE_F0_THRESH = 155.0    # Hz — below this → male
MIN_VOICED_FRAC = 0.3     # minimum voiced fraction for female classification


def classify(median_f0: float | None, voiced_frac: float) -> str:
    if median_f0 is None:
        return "ambiguous"
    if median_f0 > FEMALE_F0_THRESH and voiced_frac > MIN_VOICED_FRAC:
        return "female"
    if median_f0 < MALE_F0_THRESH:
        return "male"
    return "ambiguous"


def audit_clip(path: pathlib.Path) -> dict:
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=70.0, pitch_ceiling=400.0)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]
    total_frames = len(f0_values)
    voiced_frac = len(voiced) / total_frames if total_frames > 0 else 0.0
    median_f0 = float(np.median(voiced)) if len(voiced) > 0 else None
    mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else None
    gender = classify(median_f0, voiced_frac)
    return {
        "clip_id": path.stem,
        "median_f0_hz": round(median_f0, 1) if median_f0 is not None else "",
        "mean_f0_hz": round(mean_f0, 1) if mean_f0 is not None else "",
        "voiced_fraction": round(voiced_frac, 3),
        "gender_guess": gender,
    }


def main() -> None:
    clips = sorted(CLIPS_DIR.glob("*.wav"))
    if not clips:
        print(f"ERROR: no .wav files found in {CLIPS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Auditing {len(clips)} clips...", flush=True)
    rows: list[dict] = []
    for i, clip in enumerate(clips):
        row = audit_clip(clip)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(clips)}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")

    # --- histogram ---
    median_vals = [float(r["median_f0_hz"]) for r in rows if r["median_f0_hz"] != ""]
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(median_vals, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(165, color="red", linestyle="--", label="165 Hz divider")
    ax.axvline(MALE_F0_THRESH, color="orange", linestyle=":", label=f"{MALE_F0_THRESH} Hz male thresh")
    ax.axvline(FEMALE_F0_THRESH, color="green", linestyle=":", label=f"{FEMALE_F0_THRESH} Hz female thresh")
    ax.set_xlabel("Median F0 (Hz)")
    ax.set_ylabel("Clip count")
    ax.set_title(f"BBC clips — median F0 distribution (n={len(median_vals)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Wrote {OUT_PLOT}")

    # --- summary ---
    counts = {"male": 0, "female": 0, "ambiguous": 0}
    for r in rows:
        counts[r["gender_guess"]] += 1

    n_total = len(rows)
    n_female = counts["female"]
    frac = n_female / n_total if n_total > 0 else 0.0

    print("\n=== BBC speaker audit summary ===")
    print(f"  male:      {counts['male']}")
    print(f"  female:    {counts['female']}")
    print(f"  ambiguous: {counts['ambiguous']}")
    print(f"  n_total:   {n_total}")
    print(f"  female fraction: {frac:.3f}")

    print("\n=== Decision rule ===")
    if frac < 0.10:
        print("  → BBC is essentially male. Keep all BBC in modern_rp pool.")
        decision = "keep_all"
    elif frac < 0.30:
        print("  → BBC mostly male but contaminated. Use gender_guess=='male' subset only.")
        decision = "filter_male"
    else:
        print("  → BBC not usable for male-target norms. Drop BBC entirely.")
        decision = "drop"
    print(f"  decision: {decision}")


if __name__ == "__main__":
    main()
