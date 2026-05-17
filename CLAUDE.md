# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Goal

Build a voice model that imitates Benedict Cumberbatch's narrator voice from his audiobooks. Two viable paths (see "Approach" below): zero-shot inference with a modern reference-based TTS, or fine-tuning on the local dataset.

## Hardware

- macOS, **Apple M3 Pro, 18 GB unified memory**
- Use **MPS** (`torch.device("mps")`) for inference; CPU fallback only when an op isn't MPS-supported
- **Do not fine-tune large TTS models locally** — 18 GB is at the edge for XTTS-v2 / StyleTTS2 training. For real fine-tunes, rent a 24 GB GPU (RunPod/Colab) and only do inference on the M3.

## Layout

| Path | Contents |
|---|---|
| [raw_cumberbatch_data/](raw_cumberbatch_data/) | 5 source audiobooks (~10 GB WAVs): Casanova, Scales of Justice, Artists in Crime, Metamorphosis, Sherlock Holmes |
| [dataset/chunks/](dataset/chunks/) | 1872 × 30s WAV chunks (~2.3 GB). **Known issue: cuts mid-sentence/mid-word** — needs re-segmentation before training |
| [dataset/transcriptions.csv](dataset/transcriptions.csv) | Whisper transcripts, pipe-delimited (`file_name\|transcription`). **Known issue: proper-noun errors** (e.g. "MEMOISE" → MEMOIRS, "Sam Simele" → San Samuele) |
| [fine-tune-model-claude.ipynb](fine-tune-model-claude.ipynb) | Old Tacotron2/Coqui-TTS fine-tune pipeline. **Deprecated** — Coqui is unmaintained since 2023 and Tacotron2 is obsolete |
| [generate_voice.ipynb](generate_voice.ipynb) | XTTS zero-shot generation playground |
| [generate_subs_from_folder.ipynb](generate_subs_from_folder.ipynb) | Whisper transcription pipeline that produced the 30s chunks |
| [cumberbatch_xtts_embedded.wav](cumberbatch_xtts_embedded.wav) | Earlier XTTS-v2 zero-shot output |
| [scripts/](scripts/) | Analysis, dataset, and fine-tune scripts (see below) |
| [pyproject.toml](pyproject.toml) | uv project config — single venv with F5-TTS, Whisper large-v3, silero-VAD, accelerate, datasets |
| `.venv/` | Python 3.10 venv (managed by uv via `uv sync`) |

## Book chunk ranges

The 1872 chunks are sequential across all 5 audiobooks:

| Book | Chunks | Hours | Style notes |
|---|---|---|---|
| Casanova | 000–605 | 5.0 h | **Most consistent narrator style** (rms_cv=0.178) — first-person memoir, uniform reading voice |
| Scales of Justice | 606–1055 | 3.8 h | Most inconsistent (rms_cv=0.304) — heavy detective dialogue, avoid for fine-tuning |
| Artists in Crime | 1056–1423 | 3.1 h | Moderate consistency; studio/art-world narration |
| Metamorphosis | 1426–1622 | 1.6 h | Very quiet (rms_mean=0.026), subdued literary style; consistent but low energy |
| Sherlock Holmes | 1623–1871 | 2.1 h | Moderate; lots of Holmes/Watson character acting |

**Source**: measured by `scripts/analyze_books.py` across all 1872 chunks using MFCC+RMS embeddings + KMeans clustering. Lower rms_cv = more consistent volume/energy = safer for single-style fine-tune.

## Dataset caveats (read before touching training code)

1. **30 s fixed chunks are wrong for TTS training.** Modern voice models want 2–12 s clips split on **silence/sentence boundaries**, not Whisper's transcription window. Re-segment with silero-VAD or pyannote before any fine-tune.
2. **Audiobook is multi-role.** Cumberbatch performs characters (Holmes, Moriarty, women's voices, dialects). A single-speaker fine-tune trained on the whole dataset produces mush. Before fine-tuning, **keep narrator-only segments** (see dataset pipeline below), or generate with a zero-shot model and supply a clean narrator reference clip.
3. **Whisper transcripts need cleanup.** Proper nouns and archaic words are mis-transcribed (e.g. "MEMOISE" → MEMOIRS). Re-transcribe with `whisper large-v3` which handles archaic English better.
4. **Start with Casanova chunks (000–605) for fine-tuning** — measured as the most consistent narrator style. Avoid Scales of Justice.

## Scripts

All scripts run from the single uv-managed venv. Activate with `uv run python <script>` or `source .venv/bin/activate`.

| Script | Purpose |
|---|---|
| `scripts/analyze_books.py` | Per-book style variance ranking (already ran) |
| `scripts/analyze_style.py` | Cluster segments by voice style across all chunks |
| `scripts/analyze_casanova.py` | Deep style clustering on Casanova-only — confirmed uniform |
| `scripts/segment_vad.py` | Re-segment raw audiobook via ffmpeg + silero-VAD (memory-efficient streaming) |
| `scripts/transcribe_v3.py` | Re-transcribe segments with Whisper large-v3 |
| `scripts/build_manifest.py` | Build filtered training manifest |
| `scripts/prepare_f5_dataset.py` | Convert manifest into F5-TTS dataset format (metadata.csv + wavs/) |
| `scripts/finetune_f5.py` | F5-TTS fine-tune loop on MPS — proves loss decreases |
| `scripts/run_pipeline.sh` | End-to-end orchestrator: segment → transcribe → manifest → prepare → finetune |
| `scripts/f5_infer.py` | F5-TTS zero-shot inference (MPS) |

## Approach

**Path A — zero-shot (try first, no training):**
- **F5-TTS** is the preferred zero-shot model — runs on MPS, higher quality than XTTS-v2 for cloning
- XTTS-v2 also available but must run on CPU (MPS fails: `Output channels > 65536 not supported`)
- Use a 10–20 s clean Casanova narrator clip as reference (e.g. chunk 067.wav first 12 s)
- XTTS-v2 model is cached at `~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2/`
- Run: `source .venv-f5/bin/activate && python scripts/f5_infer.py --ref tts_output/ref_narrator.wav --ref-text "..." --text "..."`

**Path B — fine-tune (for max fidelity):**

Dataset pipeline (run in order; everything but step 5 is built):

```bash
# 1. Re-segment Casanova at silence boundaries (3–12 s clips)
source .venv/bin/activate
python scripts/segment_vad.py --book casanova
# → dataset/segments/casanova/seg_*.wav + manifest.csv

# 2. Re-transcribe with Whisper large-v3 (better proper-noun handling)
python scripts/transcribe_v3.py --book casanova
# → dataset/segments/casanova/transcripts.csv

# 3. Build filtered training manifest
python scripts/build_manifest.py --book casanova
# → dataset/segments/casanova/training_manifest.csv
```

**About narrator filtering**: deep clustering of Casanova (`analyze_casanova.py`) showed
the 5 sub-clusters are all in a narrow RMS band (0.030–0.042) and all contain first-person
narrator content. Casanova is genuinely uniform — **filtering character voices is not
needed for this book**. The intra-book consistency holds at the segment level too.

5. **(Not built yet)** Fine-tune XTTS-v2 or StyleTTS2 on a rented 24 GB GPU
   (RunPod/Colab); inference locally on MPS. Do not revive the Tacotron2/Coqui path.

## Environment

Single uv-managed venv:

```bash
uv sync                    # one-time: installs everything from pyproject.toml into .venv
uv run python <script>     # run anything (auto-activates the venv)

# F5-TTS has a startup quirk on uv venvs — set this if you see a PYTHONHASHSEED crash:
PYTHONHASHSEED=random uv run python scripts/finetune_f5.py
```

Note: `requirements.txt` is pinned for the old Tacotron2 setup — ignore it, use `pyproject.toml` via uv.

## Memory constraints (M3 Pro 18 GB)

Lessons learned:
- **Do not load the 3.3 GB Casanova WAV into Python directly** — OOM-kills silently. Use `ffmpeg` to extract 16 kHz mono for VAD (drops to ~150 MB) and stream segment slices back via ffmpeg.
- F5-TTS spawns subprocesses that read `PYTHONHASHSEED` at startup; if it's set to empty string (uv-venv quirk), Python crashes. Scripts that invoke F5-TTS self-correct via `os.execv`.
- Whisper large-v3 needs ~3 GB. Don't run it concurrently with F5-TTS fine-tuning.

## Conventions

- Code runs **locally** on the user's Mac (M3 Pro).
- Prefer modifying existing notebooks over creating new ones; only create new notebooks when the existing flow doesn't fit.
- Do not commit `.env`, `dataset/`, `raw_cumberbatch_data/`, `tts_output/`, `.venv/`, `.venv-f5/` — gitignored.
