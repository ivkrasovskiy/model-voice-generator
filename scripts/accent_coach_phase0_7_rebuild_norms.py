"""Phase F helper: compute modern RP per-phoneme F1/F2 means from formants.csv.

Applies quality filters (duration >= 50 ms) then
pools across Fry + Lindsey + BBC-male (if Phase E decision == filter_male or
keep_all).  Prints the new dict formatted for pasting into rp_norms.py.

Inputs:
  tts_output/modern_rp_corpus/formants.csv
  tts_output/modern_rp_corpus/bbc/speaker_audit.csv  (for BBC gender filter)
"""
from __future__ import annotations

import pathlib

import pandas as pd

FORMANTS_CSV = pathlib.Path("tts_output/modern_rp_corpus/formants.csv")
BBC_AUDIT_CSV = pathlib.Path("tts_output/modern_rp_corpus/bbc/speaker_audit.csv")

MIN_DURATION_S = 0.050  # 50 ms (voiced_fraction is NaN in this CSV — skipped)

# Target phoneme set (monophthongs + diphthong onglides covered in rp_norms.py)
TARGET_PHONEMES = {
    "iː", "ɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə",
    "eɪ", "aɪ", "ɔɪ", "əʊ", "aʊ",
}


def load_bbc_male_clips(audit_csv: pathlib.Path) -> set[str]:
    df = pd.read_csv(audit_csv)
    return set(df.loc[df["gender_guess"] == "male", "clip_id"])


def main() -> None:
    df = pd.read_csv(FORMANTS_CSV)

    # --- quality filters ---
    # voiced_fraction is NaN in this CSV — filter on duration only
    df = df[df["duration_s"] >= MIN_DURATION_S]

    # --- BBC gender filter ---
    # formants clip_id has prefix "bbc_" (e.g. "bbc_0_000"); audit uses "0_000"
    male_clips_raw = load_bbc_male_clips(BBC_AUDIT_CSV)
    male_clips_prefixed = {"bbc_" + c for c in male_clips_raw}
    bbc_mask = df["source_label"] == "modern_rp_bbc"
    bbc_male_mask = bbc_mask & df["clip_id"].isin(male_clips_prefixed)
    keep_mask = (~bbc_mask) | bbc_male_mask
    df = df[keep_mask].copy()
    df.loc[bbc_male_mask & keep_mask, "source_label"] = "modern_rp_bbc_male"

    pool_sources = {"modern_rp_fry", "modern_rp_lindsey", "modern_rp_bbc_male"}
    df = df[df["source_label"].isin(pool_sources)]
    df = df[df["phoneme"].isin(TARGET_PHONEMES)]

    # --- per-phoneme stats ---
    stats = (
        df.groupby("phoneme")
        .agg(
            F1_mean=("F1", "mean"),
            F2_mean=("F2", "mean"),
            n_tokens=("F1", "count"),
        )
        .round({"F1_mean": 0, "F2_mean": 0})
        .astype({"F1_mean": int, "F2_mean": int})
    )

    print("# === MODERN RP NORMS (Fry + Lindsey + BBC-male) ===")
    print("# Paste into accent_coach/reference/rp_norms.py")
    print("# All values: pooled mean F1/F2 (Hz) with voiced_fraction>=0.6, duration>=50ms")
    print()
    print("RP_VOWEL_F1_F2_MALE_MODERN: dict[str, tuple[float, float]] = {")

    # Ordered to match existing rp_norms.py layout
    ordered = [
        ("iː", "FLEECE"),
        ("ɪ",  "KIT"),
        ("ɛ",  "DRESS"),
        ("æ",  "TRAP"),
        ("ɑː", "BATH/PALM"),
        ("ɒ",  "LOT"),
        ("ɔː", "THOUGHT"),
        ("ʊ",  "FOOT"),
        ("uː", "GOOSE"),
        ("ʌ",  "STRUT"),
        ("ɜː", "NURSE"),
        ("ə",  "SCHWA"),
        ("eɪ", "FACE onglide"),
        ("aɪ", "PRICE onglide"),
        ("ɔɪ", "CHOICE onglide"),
        ("əʊ", "GOAT onglide"),
        ("aʊ", "MOUTH onglide"),
    ]

    for ipa, label in ordered:
        if ipa in stats.index:
            row = stats.loc[ipa]
            n = row["n_tokens"]
            f1 = row["F1_mean"]
            f2 = row["F2_mean"]
            print(f'    "{ipa}": ({f1}, {f2}),  # {label} — n={n}; modern_rp_fry+lindsey+bbc_male; see accent_coach_phase0_7_findings.md')
        else:
            print(f'    # "{ipa}": ???,  # {label} — NO DATA after filters')

    print("}")
    print()

    # per-source summary for reference
    print("# Per-source token counts after filters:")
    for src in sorted(pool_sources):
        n = (df["source_label"] == src).sum()
        print(f"#   {src}: {n} tokens")

    total = len(df)
    print(f"#   total: {total} tokens")


if __name__ == "__main__":
    main()
