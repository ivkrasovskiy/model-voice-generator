"""
Per-book style consistency analysis.

Maps chunks to their source book, then measures how consistent (low-variance)
the voice style is within each book. The book with the lowest intra-book RMS
variance is the safest choice for a single-style fine-tune.

Usage:
    source .venv/bin/activate
    python scripts/analyze_books.py

Output:
    dataset/style_analysis/book_report.txt   — ranked book summary
    dataset/style_analysis/book_clusters/    — per-book PCA plots
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

CHUNKS_DIR = Path("dataset/chunks")
CSV_PATH = Path("dataset/transcriptions.csv")
OUT_DIR = Path("dataset/style_analysis")
OUT_DIR.mkdir(exist_ok=True)

# Book boundaries derived from transcription markers (chunk number ranges, inclusive)
BOOKS = {
    "Casanova":          (0,    605),
    "Scales_of_Justice": (606,  1055),
    "Artists_in_Crime":  (1056, 1423),
    "Metamorphosis":     (1426, 1622),
    "Sherlock_Holmes":   (1623, 1871),
}


def mfcc_embedding(wav_path: Path) -> np.ndarray:
    """Fast MFCC+pitch embedding — same as analyze_style.py."""
    audio, sr = librosa.load(wav_path, sr=16000, mono=True, duration=10)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_vals = pitch[pitch > 0]
    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [np.mean(pitch_vals) if len(pitch_vals) else 0],
        [np.std(pitch_vals) if len(pitch_vals) else 0],
        [librosa.feature.rms(y=audio).mean()],
        [librosa.feature.spectral_centroid(y=audio, sr=sr).mean()],
        [librosa.feature.zero_crossing_rate(audio).mean()],
    ])


def analyze_book(name: str, start: int, end: int, df: pd.DataFrame) -> dict:
    """Compute style stats for one book's chunk range."""
    book_df = df[(df["num"] >= start) & (df["num"] <= end)]
    chunks = [CHUNKS_DIR / row["chunk"] for _, row in book_df.iterrows()
              if (CHUNKS_DIR / row["chunk"]).exists()]

    if not chunks:
        return {}

    embeddings = []
    rms_vals = []
    for wav in tqdm(chunks, desc=name, leave=False):
        try:
            emb = mfcc_embedding(wav)
            embeddings.append(emb)
            audio, sr = librosa.load(wav, sr=16000, mono=True, duration=10)
            rms_vals.append(float(librosa.feature.rms(y=audio).mean()))
        except Exception:
            pass

    embeddings = np.array(embeddings)
    rms_vals = np.array(rms_vals)

    # Intra-book style variance — lower = more consistent narrator voice
    emb_var = float(np.mean(np.var(embeddings, axis=0)))
    rms_mean = float(rms_vals.mean())
    rms_std = float(rms_vals.std())
    # Coefficient of variation (relative spread)
    rms_cv = rms_std / rms_mean if rms_mean > 0 else 0

    return {
        "book": name,
        "chunks": len(chunks),
        "duration_hrs": len(chunks) * 30 / 3600,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_cv": rms_cv,              # lower = more consistent volume
        "embedding_variance": emb_var,  # lower = more consistent style overall
    }


def main():
    df = pd.read_csv(CSV_PATH, delimiter="|")
    df.columns = ["chunk", "text"]
    df["num"] = df["chunk"].str.replace(".wav", "", regex=False).astype(int)

    results = []
    for name, (start, end) in BOOKS.items():
        print(f"\nAnalyzing {name} (chunks {start}–{end})...")
        stats = analyze_book(name, start, end, df)
        if stats:
            results.append(stats)
            print(f"  chunks={stats['chunks']}  duration={stats['duration_hrs']:.1f}h  "
                  f"rms_cv={stats['rms_cv']:.3f}  style_var={stats['embedding_variance']:.4f}")

    results_df = pd.DataFrame(results).sort_values("rms_cv")

    report = [
        "=== Per-Book Style Consistency ===",
        "Lower rms_cv and style_var = more consistent narrator voice",
        "",
        f"{'Book':<25} {'Chunks':>7} {'Hours':>6} {'RMS mean':>9} {'RMS cv':>8} {'Style var':>10}",
        "-" * 70,
    ]
    for _, row in results_df.iterrows():
        marker = " ← most consistent" if row["book"] == results_df.iloc[0]["book"] else ""
        report.append(
            f"{row['book']:<25} {int(row['chunks']):>7} {row['duration_hrs']:>6.1f} "
            f"{row['rms_mean']:>9.4f} {row['rms_cv']:>8.3f} {row['embedding_variance']:>10.4f}"
            + marker
        )

    report += [
        "",
        "Recommendation:",
        "  1. Pick the book with lowest rms_cv (most consistent volume/energy).",
        "  2. Run analyze_style.py on just that book's chunk range to find",
        "     the dominant narrator cluster within it.",
        "  3. Keep only dominant-cluster chunks for fine-tuning.",
    ]

    report_text = "\n".join(report)
    print("\n" + report_text)
    (OUT_DIR / "book_report.txt").write_text(report_text)
    print(f"\nSaved → {OUT_DIR}/book_report.txt")


if __name__ == "__main__":
    main()
