"""Phase 0.8: vowel-space normalisation hypotheses.

Runs H1 (VTL) → H3 (anchor) → H2 (Nearey) → H4 (Bark) in order.
Stops at the first GREEN (all four cluster criteria pass) or after all
four hypotheses if none pass.

Outputs:
    tts_output/accent_coach/phase0_8/h1_vtl.json
    tts_output/accent_coach/phase0_8/h3_anchor.json
    tts_output/accent_coach/phase0_8/h2_nearey.json
    tts_output/accent_coach/phase0_8/h4_bark.json
    tts_output/accent_coach/phase0_8/cluster_metrics_summary.csv
    tts_output/accent_coach/phase0_8/per_phoneme_distances_<winner>.csv  (if GREEN)
    tts_output/accent_coach/phase0_8/piecewise_scores_<winner>.json      (if GREEN)
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import pathlib
import random
import sys

import numpy as np
import parselmouth

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from accent_coach.comparison.vowels import score_vowels_piecewise
from accent_coach.diagnostics.anchor_normalize import anchor_normalize
from accent_coach.diagnostics.bark_distance import bark_transform
from accent_coach.diagnostics.cluster_eval import evaluate_clustering, per_phoneme_sigma_rp
from accent_coach.diagnostics.nearey_normalize import nearey_corner_normalize
from accent_coach.diagnostics.vtl_normalize import vtl_normalize
from accent_coach.reference.rp_norms import RP_VOWEL_F1_F2_MALE_LEGACY

OUT_DIR = pathlib.Path("tts_output/accent_coach/phase0_8")
CENTROIDS_PATH = pathlib.Path("tts_output/accent_coach/bench/phase0_7/speaker_centroids.json")

CLIPS = {
    "fry": pathlib.Path("tts_output/modern_rp_corpus/fry/clips"),
    "lindsey": pathlib.Path("tts_output/modern_rp_corpus/lindsey/clips"),
    "bbc_male": pathlib.Path("tts_output/modern_rp_corpus/bbc/clips"),
    "real_BC": pathlib.Path("tts_output/real_bc_corpus/clips"),
    "owner": pathlib.Path("tts_output/accent_coach/users/owner"),
}
BBC_AUDIT_CSV = pathlib.Path("tts_output/modern_rp_corpus/bbc/speaker_audit.csv")

N_SAMPLE = 20  # clips per group for F0 estimation


# ---------------------------------------------------------------------------
# F0 estimation helpers
# ---------------------------------------------------------------------------

def estimate_f0_from_clips(clips_dir: pathlib.Path, n: int = N_SAMPLE) -> float | None:
    """Return mean voiced F0 across a random sample of WAV clips."""
    wavs = list(clips_dir.glob("*.wav"))
    if not wavs:
        return None
    sample = random.sample(wavs, min(n, len(wavs)))
    voiced_f0s: list[float] = []
    for wav in sample:
        try:
            snd = parselmouth.Sound(str(wav))
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=70.0, pitch_ceiling=400.0)
            vals = pitch.selected_array["frequency"]
            voiced_f0s.extend(float(v) for v in vals if v > 0)
        except Exception:
            continue
    return float(np.mean(voiced_f0s)) if voiced_f0s else None


def bbc_male_f0_from_audit(audit_csv: pathlib.Path) -> float:
    """Read precomputed BBC male mean F0 from the Phase E audit CSV."""
    f0s: list[float] = []
    with open(audit_csv) as fh:
        for row in csv.DictReader(fh):
            if row["gender_guess"] == "male" and row["mean_f0_hz"]:
                with contextlib.suppress(ValueError):
                    f0s.append(float(row["mean_f0_hz"]))
    return float(np.mean(f0s)) if f0s else 124.6  # fallback to Phase E measurement


def estimate_all_f0s() -> dict[str, float]:
    """Return mean F0 per speaker group needed for H1."""
    print("  Estimating F0 per speaker group …")
    f0s: dict[str, float] = {}

    # BBC male: use precomputed audit (fast)
    f0s["bbc_male"] = bbc_male_f0_from_audit(BBC_AUDIT_CSV)
    print(f"    bbc_male: {f0s['bbc_male']:.1f} Hz (from audit CSV)")

    # Other groups: sample clips
    for spk in ("fry", "lindsey", "real_BC", "owner"):
        d = CLIPS.get(spk)
        if d is None or not d.exists():
            print(f"    {spk}: clips dir missing — using 120 Hz fallback")
            f0s[spk] = 120.0
            continue
        val = estimate_f0_from_clips(d)
        if val is None:
            print(f"    {spk}: no voiced frames found — using 120 Hz fallback")
            f0s[spk] = 120.0
        else:
            f0s[spk] = val
            print(f"    {spk}: {val:.1f} Hz")

    # synth BC: TTS output, likely similar to real BC; estimate from clips
    synth_dir = pathlib.Path("tts_output/accent_coach/bc_cal_50")
    val = estimate_f0_from_clips(synth_dir) if synth_dir.exists() else None
    f0s["synth_BC"] = val if val else f0s.get("real_BC", 120.0)
    print(f"    synth_BC: {f0s['synth_BC']:.1f} Hz")

    # modern_rp centroid: geometric mean of fry/lindsey/bbc_male
    f0s["modern_rp"] = (f0s["fry"] * f0s["lindsey"] * f0s["bbc_male"]) ** (1 / 3)
    print(f"    modern_rp (geometric mean): {f0s['modern_rp']:.1f} Hz")

    # deterding: literature table → assign F0_ref so no correction applied
    f0_ref = f0s["modern_rp"]
    f0s["deterding"] = f0_ref
    print(f"    deterding: {f0s['deterding']:.1f} Hz (= F0_ref, no correction)")

    return f0s


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_centroids() -> dict:
    """Load speaker_centroids.json and add deterding pseudo-speaker."""
    with open(CENTROIDS_PATH) as fh:
        raw = json.load(fh)

    # Add deterding as a speaker from the legacy norms
    for phoneme, f1f2 in RP_VOWEL_F1_F2_MALE_LEGACY.items():
        if phoneme not in raw:
            raw[phoneme] = {}
        raw[phoneme]["deterding"] = {"f1": f1f2[0], "f2": f1f2[1], "n": 0}

    # Normalise speaker key: the JSON uses "real_BC" — keep as-is
    return raw


def centroids_f1f2(centroids: dict, spk: str) -> dict[str, tuple[float, float]]:
    """Extract {phoneme: (f1, f2)} for a single speaker from centroids."""
    result: dict[str, tuple[float, float]] = {}
    for phoneme, speakers in centroids.items():
        v = speakers.get(spk)
        if v and v["f1"] is not None and v["f2"] is not None:
            result[phoneme] = (v["f1"], v["f2"])
    return result


# ---------------------------------------------------------------------------
# Hypothesis runner
# ---------------------------------------------------------------------------

def run_hypothesis(
    name: str,
    label: str,
    normalised: dict,
    winner_so_far: str | None,
    summary_rows: list[dict],
) -> tuple[dict, str | None]:
    """Evaluate clustering on `normalised` centroids; write JSON; return (result, winner)."""
    result = evaluate_clustering(normalised)
    result["hypothesis"] = name

    out_path = OUT_DIR / f"{label}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"\n  {name} ({label})")
    print(f"    C1 within-RP = {result['C1_within_rp_mean_dist']:.1f}  (pass ≤ 50: {result['C1_pass']})")
    print(f"    C2 RP vs Deterding = {result['C2_rp_vs_deterding']:.1f}  ratio={result['C2_ratio']:.2f}  (pass ≥1.5×: {result['C2_pass']})")
    print(f"    C3 owner vs RP     = {result['C3_owner_vs_rp']:.1f}    ratio={result['C3_ratio']:.2f}  (pass ≥2×: {result['C3_pass']})")
    print(f"    C4 owner vs Deter  = {result['C4_owner_vs_deterding']:.1f}  ratio={result['C4_ratio']:.2f}  (pass ≥1.5×: {result['C4_pass']})")
    print(f"    overall_pass = {result['overall_pass']}")

    summary_rows.append({
        "hypothesis": name,
        "C1": result["C1_within_rp_mean_dist"],
        "C2": result["C2_rp_vs_deterding"],
        "C3": result["C3_owner_vs_rp"],
        "C4": result["C4_owner_vs_deterding"],
        "C2_ratio": result["C2_ratio"],
        "C3_ratio": result["C3_ratio"],
        "C4_ratio": result["C4_ratio"],
        "C1_pass": result["C1_pass"],
        "C2_pass": result["C2_pass"],
        "C3_pass": result["C3_pass"],
        "C4_pass": result["C4_pass"],
        "overall_pass": result["overall_pass"],
    })

    new_winner = winner_so_far
    if result["overall_pass"] and new_winner is None:
        new_winner = label
    return result, new_winner


def run_piecewise_scores(
    label: str,
    normalised: dict,
    speakers: list[str] = ("real_BC", "synth_BC", "owner"),
) -> dict:
    """Compute piecewise vowel scores for each speaker vs modern_rp."""
    sigma_rp = per_phoneme_sigma_rp(normalised)
    ref_f1f2 = centroids_f1f2(normalised, "modern_rp")

    scores: dict[str, dict] = {}
    for spk in speakers:
        user_f1f2 = centroids_f1f2(normalised, spk)
        scores[spk] = score_vowels_piecewise(user_f1f2, ref_f1f2, sigma_rp)

    out = {"hypothesis": label, "sigma_rp": {k: round(v, 2) for k, v in sigma_rp.items()}, "scores": scores}
    out_path = OUT_DIR / f"piecewise_scores_{label}.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n  Piecewise scores ({label}):")
    for spk, sc in scores.items():
        print(f"    {spk}: composite={sc['composite']}")
    return out


def write_per_phoneme_csv(label: str, normalised: dict) -> None:
    """Write per-phoneme F1/F2 and distance-to-RP for all speakers."""
    out_path = OUT_DIR / f"per_phoneme_distances_{label}.csv"
    rows: list[dict] = []
    speakers = ["real_BC", "synth_BC", "fry", "lindsey", "bbc_male", "owner", "deterding"]
    for phoneme, spk_dict in normalised.items():
        ref = spk_dict.get("modern_rp")
        row: dict = {"phoneme": phoneme}
        for spk in speakers:
            v = spk_dict.get(spk)
            if v:
                row[f"{spk}_f1"] = round(v["f1"], 3)
                row[f"{spk}_f2"] = round(v["f2"], 3)
                if ref:
                    d = math.sqrt((v["f1"] - ref["f1"]) ** 2 + (v["f2"] - ref["f2"]) ** 2)
                    row[f"{spk}_dist_rp"] = round(d, 3)
            else:
                row[f"{spk}_f1"] = ""
                row[f"{spk}_f2"] = ""
                row[f"{spk}_dist_rp"] = ""
        rows.append(row)

    if rows:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 0.8: vowel-space normalisation hypotheses ===\n")
    print("Loading centroids …")
    centroids = load_centroids()
    print(f"  Phonemes: {list(centroids.keys())}")
    print(f"  Speakers: {set(s for d in centroids.values() for s in d)}")

    # F0 estimation (needed only for H1)
    print("\nEstimating F0 per speaker group (for H1 VTL scaling) …")
    f0s = estimate_all_f0s()

    summary_rows: list[dict] = []
    winner: str | None = None

    # ---- H1: VTL scaling ----
    print("\n--- H1: VTL scaling ---")
    h1_norm = vtl_normalize(centroids, f0s)
    _, winner = run_hypothesis("H1 — VTL scaling", "h1_vtl", h1_norm, winner, summary_rows)
    if winner:
        print("\n  >> GREEN — stopping at H1")

    # ---- H3: Anchor-relative (primary candidate) ----
    if winner is None:
        print("\n--- H3: Anchor-relative coordinates ---")
        h3_norm = anchor_normalize(centroids)
        _, winner = run_hypothesis("H3 — Anchor-relative", "h3_anchor", h3_norm, winner, summary_rows)
        if winner:
            print("\n  >> GREEN — stopping at H3")

    # ---- H2: Nearey log-mean ----
    if winner is None:
        print("\n--- H2: Nearey log-mean ---")
        h2_norm = nearey_corner_normalize(centroids)
        _, winner = run_hypothesis("H2 — Nearey log-mean", "h2_nearey", h2_norm, winner, summary_rows)
        if winner:
            print("\n  >> GREEN — stopping at H2")

    # ---- H4: Bark distance ----
    if winner is None:
        print("\n--- H4: Bark-scale ---")
        h4_norm = bark_transform(centroids)
        _, winner = run_hypothesis("H4 — Bark scale", "h4_bark", h4_norm, winner, summary_rows)
        if winner:
            print("\n  >> GREEN — stopping at H4")

    # ---- Summary CSV ----
    summary_path = OUT_DIR / "cluster_metrics_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")

    # ---- Piecewise scores + per-phoneme CSV for winner (or best partial) ----
    if winner:
        print(f"\n=== Winner: {winner} ===")
        norm_map = {
            "h1_vtl": h1_norm if winner == "h1_vtl" else None,
            "h3_anchor": h3_norm if winner == "h3_anchor" else None,
            "h2_nearey": h2_norm if winner == "h2_nearey" else None,
            "h4_bark": h4_norm if winner == "h4_bark" else None,
        }
        winning_norm = norm_map[winner]
        if winning_norm:
            run_piecewise_scores(winner, winning_norm)
            write_per_phoneme_csv(winner, winning_norm)
    else:
        print("\nNo hypothesis passed all four criteria.")
        # Still write piecewise scores for all run hypotheses so we can compare
        print("Running piecewise scores for all hypotheses (no winner — diagnostic mode) …")
        # Rebuild norm map from what was computed
        hyp_norms: dict[str, dict] = {}
        for row in summary_rows:
            lbl_raw = row["hypothesis"]
            if "VTL" in lbl_raw:
                hyp_norms["h1_vtl"] = h1_norm
            elif "Anchor" in lbl_raw:
                hyp_norms["h3_anchor"] = h3_norm
            elif "Nearey" in lbl_raw:
                hyp_norms["h2_nearey"] = h2_norm
            elif "Bark" in lbl_raw:
                hyp_norms["h4_bark"] = h4_norm

        for lbl, norm in hyp_norms.items():
            run_piecewise_scores(lbl, norm)
            write_per_phoneme_csv(lbl, norm)

        # Determine verdict
        all_c1_fail = all(not r["C1_pass"] for r in summary_rows)
        all_c3_fail = all(not r["C3_pass"] for r in summary_rows)

        if all_c1_fail:
            print("\n>>> RED — C1 fails for all hypotheses. Data quality problem; halt.")
        elif all_c3_fail:
            print("\n>>> RED (or YELLOW) — C3 fails for all hypotheses (owner indistinguishable from modern RP).")
        else:
            best_pass_count = max(
                sum([r["C1_pass"], r["C2_pass"], r["C3_pass"], r["C4_pass"]])
                for r in summary_rows
            )
            print(f"\n>>> YELLOW — best hypothesis passes {best_pass_count}/4 criteria. Consult owner.")

    print("\nDone.")


if __name__ == "__main__":
    main()
