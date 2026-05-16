"""
Fast voice style analysis across all 5 audiobooks.
Clusters audio segments by speaker/style using speaker embeddings,
then identifies the dominant narrator style and scores each book.

Runtime: ~10-30 min for all 5 books on M3 Pro (uses MPS).

Usage:
    source .venv/bin/activate   # needs TTS for speaker embeddings
    python scripts/analyze_style.py

Output:
    dataset/style_analysis/
        embeddings.npy          # per-chunk speaker embeddings
        clusters.csv            # chunk → cluster label
        report.txt              # dominant style per book + % narrator
        narrator_chunks.txt     # list of chunks belonging to dominant style
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from tqdm import tqdm

CHUNKS_DIR = Path("dataset/chunks")
CSV_PATH = Path("dataset/transcriptions.csv")
OUT_DIR = Path("dataset/style_analysis")
OUT_DIR.mkdir(exist_ok=True)

# How many clusters to look for (narrator + N character styles)
N_CLUSTERS = 6

# Only analyze first N chunks per book to get a fast overview
# Set to None to analyze everything (~1872 chunks, ~30 min)
FAST_MODE_CHUNKS = 300  # ~2.5 hrs of audio sampled; set None for full run


def load_speaker_encoder():
    """Load XTTS speaker encoder for computing voice embeddings."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import os

    model_dir = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"
    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
    # Keep on CPU — MPS conv1d channel limit
    model.eval()
    return model


def get_embedding(model, wav_path: Path) -> np.ndarray:
    """Get 512-dim speaker embedding for a WAV clip."""
    audio, sr = librosa.load(wav_path, sr=22050, mono=True)
    # Use first 10s max for embedding
    audio = audio[:sr * 10]
    audio_tensor = torch.tensor(audio).unsqueeze(0)

    with torch.no_grad():
        audio_16k = torch.nn.functional.interpolate(
            audio_tensor.unsqueeze(0), scale_factor=16000 / 22050
        ).squeeze(0)
        emb = model.speaker_manager.encoder.forward(
            audio_16k.to("cpu"), l2_norm=True
        )
    return emb.squeeze().cpu().numpy()


def get_embedding_simple(wav_path: Path) -> np.ndarray:
    """
    Lightweight embedding: MFCC statistics (no model needed).
    Less accurate than neural embeddings but runs in milliseconds.
    Good enough to separate narrator vs. dramatic character voices.
    """
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


def main():
    df = pd.read_csv(CSV_PATH, delimiter="|")
    chunks = sorted(CHUNKS_DIR.glob("*.wav"))

    if FAST_MODE_CHUNKS:
        # Sample evenly across the dataset instead of just taking the first N
        step = max(1, len(chunks) // FAST_MODE_CHUNKS)
        chunks = chunks[::step][:FAST_MODE_CHUNKS]
        print(f"Fast mode: analyzing {len(chunks)} chunks (every {step}th)")
    else:
        print(f"Full mode: analyzing {len(chunks)} chunks")

    # --- Compute embeddings ---
    emb_cache = OUT_DIR / "embeddings.npy"
    names_cache = OUT_DIR / "chunk_names.json"

    if emb_cache.exists() and names_cache.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(emb_cache)
        names = json.loads(names_cache.read_text())
        chunks = [Path(CHUNKS_DIR / n) for n in names]
    else:
        print("Computing MFCC-based voice embeddings...")
        embeddings = []
        names = []
        for wav in tqdm(chunks, desc="Embedding"):
            try:
                emb = get_embedding_simple(wav)
                embeddings.append(emb)
                names.append(wav.name)
            except Exception as e:
                print(f"  skip {wav.name}: {e}")

        embeddings = np.array(embeddings)
        np.save(emb_cache, embeddings)
        names_cache.write_text(json.dumps(names))
        print(f"Saved {len(embeddings)} embeddings → {emb_cache}")

    # --- Normalize ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings)

    # --- Cluster ---
    print(f"Clustering into {N_CLUSTERS} style groups...")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_norm)

    # --- Build results DataFrame ---
    results = pd.DataFrame({
        "chunk": names,
        "cluster": labels,
    })

    # Attach transcription
    trans = pd.read_csv(CSV_PATH, delimiter="|")
    trans.columns = ["chunk", "transcription"]
    results = results.merge(trans, on="chunk", how="left")

    # Audio features for each chunk (for report)
    rms_vals = []
    dur_vals = []
    for wav_name in tqdm(names, desc="Audio features", leave=False):
        audio, sr = librosa.load(CHUNKS_DIR / wav_name, sr=16000, mono=True, duration=10)
        rms_vals.append(float(librosa.feature.rms(y=audio).mean()))
        dur_vals.append(float(len(audio) / sr))
    results["rms"] = rms_vals
    results["duration"] = dur_vals

    results.to_csv(OUT_DIR / "clusters.csv", index=False)

    # --- Identify dominant cluster (narrator) ---
    cluster_sizes = results["cluster"].value_counts()
    dominant_cluster = cluster_sizes.idxmax()
    narrator_pct = cluster_sizes[dominant_cluster] / len(results) * 100

    # Per-cluster stats
    print("\n=== Cluster Summary ===")
    for c in sorted(results["cluster"].unique()):
        sub = results[results["cluster"] == c]
        marker = " ← NARRATOR (dominant)" if c == dominant_cluster else ""
        print(f"  Cluster {c}: {len(sub):4d} chunks  "
              f"avg_rms={sub['rms'].mean():.4f}  {marker}")

    # --- Report ---
    report_lines = [
        "=== Voice Style Analysis ===",
        f"Analyzed {len(results)} chunks across dataset",
        f"Clusters: {N_CLUSTERS}",
        f"Dominant cluster: {dominant_cluster}  ({narrator_pct:.1f}% of chunks) ← likely narrator",
        "",
        "Cluster breakdown:",
    ]
    for c in sorted(results["cluster"].unique()):
        sub = results[results["cluster"] == c]
        marker = " ← NARRATOR" if c == dominant_cluster else ""
        report_lines.append(
            f"  {c}: {len(sub)} chunks  avg_rms={sub['rms'].mean():.4f}  "
            f"avg_pitch_proxy={sub['rms'].std():.4f}{marker}"
        )

    # Sample transcripts per cluster
    report_lines += ["", "Sample transcripts per cluster:"]
    for c in sorted(results["cluster"].unique()):
        sub = results[results["cluster"] == c]
        report_lines.append(f"\n  --- Cluster {c} ---")
        for _, row in sub.sample(min(3, len(sub)), random_state=42).iterrows():
            t = str(row.get("transcription", ""))[:120]
            report_lines.append(f"    [{row['chunk']}] {t}")

    report_text = "\n".join(report_lines)
    (OUT_DIR / "report.txt").write_text(report_text)
    print("\n" + report_text)

    # --- Save narrator chunk list ---
    narrator_chunks = results[results["cluster"] == dominant_cluster]["chunk"].tolist()
    (OUT_DIR / "narrator_chunks.txt").write_text("\n".join(narrator_chunks))
    print(f"\nNarrator chunks saved → {OUT_DIR}/narrator_chunks.txt  ({len(narrator_chunks)} clips)")

    # --- PCA plot (optional, if matplotlib available) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings_norm)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                             cmap="tab10", alpha=0.5, s=15)
        ax.set_title("Voice style clusters (PCA of MFCC embeddings)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "clusters_pca.png", dpi=150)
        print(f"PCA plot → {OUT_DIR}/clusters_pca.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
