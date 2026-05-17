"""
Deep style analysis of Casanova only (chunks 000-605).

Uses a larger random sample than the broad analyze_style.py to confirm whether
Casanova is genuinely uniform or has hidden character-voice clusters.

Usage:
    source .venv/bin/activate
    python scripts/analyze_casanova.py

Output:
    dataset/style_analysis/casanova/
        embeddings.npy
        clusters.csv
        report.txt
        clusters_pca.png
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

CHUNKS_DIR = Path("dataset/chunks")
CSV_PATH = Path("dataset/transcriptions.csv")
OUT_DIR = Path("dataset/style_analysis/casanova")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASANOVA_RANGE = (0, 605)
N_CLUSTERS = 5
SAMPLE_SIZE = 400  # ~66% of Casanova chunks, larger than 300 used in broad sweep


def mfcc_embedding(wav_path: Path) -> np.ndarray:
    audio, sr = librosa.load(wav_path, sr=16000, mono=True, duration=15)
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


def main():
    rng = np.random.RandomState(42)

    df = pd.read_csv(CSV_PATH, delimiter="|")
    df.columns = ["chunk", "text"]
    df["num"] = df["chunk"].str.replace(".wav", "", regex=False).astype(int)
    casanova_df = df[df["num"].between(*CASANOVA_RANGE)].reset_index(drop=True)
    print(f"Casanova has {len(casanova_df)} chunks total")

    sample_size = min(SAMPLE_SIZE, len(casanova_df))
    sampled = casanova_df.sample(n=sample_size, random_state=42).sort_values("num").reset_index(drop=True)
    print(f"Sampling {sample_size} random chunks for analysis...")

    embeddings = []
    rms_vals = []
    durations = []
    names = []
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Embedding"):
        wav = CHUNKS_DIR / row["chunk"]
        if not wav.exists():
            continue
        try:
            emb = mfcc_embedding(wav)
            audio, sr = librosa.load(wav, sr=16000, mono=True, duration=15)
            rms_vals.append(float(librosa.feature.rms(y=audio).mean()))
            durations.append(len(audio) / sr)
            embeddings.append(emb)
            names.append(row["chunk"])
        except Exception as e:
            print(f"  skip {row['chunk']}: {e}")

    embeddings = np.array(embeddings)
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    (OUT_DIR / "chunk_names.json").write_text(json.dumps(names))

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    emb_norm = scaler.fit_transform(embeddings)

    print(f"\nClustering into {N_CLUSTERS} groups...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(emb_norm)

    results = pd.DataFrame({
        "chunk": names,
        "cluster": labels,
        "rms": rms_vals,
    })
    trans = pd.read_csv(CSV_PATH, delimiter="|")
    trans.columns = ["chunk", "transcription"]
    results = results.merge(trans, on="chunk", how="left")
    results.to_csv(OUT_DIR / "clusters.csv", index=False)

    cluster_sizes = results["cluster"].value_counts()
    dominant = cluster_sizes.idxmax()
    dominant_pct = cluster_sizes[dominant] / len(results) * 100

    report = [
        "=== Casanova-Only Style Analysis ===",
        f"Analyzed {len(results)} random chunks from {len(casanova_df)} total ({sample_size/len(casanova_df)*100:.0f}% sample)",
        f"Clusters: {N_CLUSTERS}",
        f"Dominant cluster: {dominant}  ({dominant_pct:.1f}% of chunks)",
        "",
        "Cluster breakdown (sorted by size):",
    ]
    for c in cluster_sizes.index:
        sub = results[results["cluster"] == c]
        marker = " ← dominant (narrator)" if c == dominant else ""
        report.append(
            f"  cluster {c}: {len(sub):4d} chunks ({len(sub)/len(results)*100:4.1f}%)  "
            f"rms_mean={sub['rms'].mean():.4f}  rms_std={sub['rms'].std():.4f}{marker}"
        )

    report += ["", "Sample transcripts per cluster (3 each):"]
    for c in cluster_sizes.index:
        sub = results[results["cluster"] == c]
        marker = " ← DOMINANT" if c == dominant else ""
        report.append(f"\n  --- Cluster {c}{marker} ---")
        for _, row in sub.sample(min(3, len(sub)), random_state=42).iterrows():
            t = str(row.get("transcription", ""))[:120]
            report.append(f"    [{row['chunk']}] {t}")

    report += [
        "",
        "Interpretation:",
        "  - If dominant cluster is >75% of chunks → Casanova is nearly uniform.",
        "    Filtering is barely needed; use full Casanova range minus the small",
        "    'character voice' clusters.",
        "  - If dominant is <60% → Casanova has multiple styles; you need to filter.",
        "  - Outlier clusters with high rms_mean = dramatic/character voices.",
        "  - Outlier clusters with low rms_mean = whispered/intimate passages.",
    ]

    text = "\n".join(report)
    print("\n" + text)
    (OUT_DIR / "report.txt").write_text(text)

    narrator_chunks = results[results["cluster"] == dominant]["chunk"].tolist()
    (OUT_DIR / "narrator_chunks.txt").write_text("\n".join(narrator_chunks))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_norm)
        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
        ax.set_title(f"Casanova style clusters (n={len(results)})")
        plt.colorbar(sc, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "clusters_pca.png", dpi=150)
        print(f"PCA plot → {OUT_DIR}/clusters_pca.png")
    except ImportError:
        pass

    print(f"\nNarrator chunk list → {OUT_DIR}/narrator_chunks.txt  ({len(narrator_chunks)} clips)")


if __name__ == "__main__":
    main()
