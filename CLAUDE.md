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
| [coqui-ai-TTS/](coqui-ai-TTS/) | Vendored Coqui TTS source — kept only for reference; do not extend |
| [cumberbatch_xtts_embedded.wav](cumberbatch_xtts_embedded.wav) | Earlier XTTS-v2 zero-shot output |
| `.venv/` | Python 3.10 venv |

## Dataset caveats (read before touching training code)

1. **30 s fixed chunks are wrong for TTS training.** Modern voice models want 2–12 s clips split on **silence/sentence boundaries**, not Whisper's transcription window. Re-segment with silero-VAD or pyannote before any fine-tune.
2. **Audiobook is multi-role.** Cumberbatch performs characters (Holmes, Moriarty, women's voices, dialects). A single-speaker fine-tune trained on the whole dataset produces mush. Before fine-tuning, **diarize and keep narrator-only segments**, or generate with a zero-shot model and supply a clean narrator reference clip.
3. **Whisper transcripts need cleanup.** Proper nouns and archaic words are mis-transcribed. Either re-transcribe with `whisper large-v3` (better proper-noun handling) or hand-correct.

## Approach

**Path A — zero-shot (try first, no training):**
- F5-TTS or XTTS-v2 with a 6–30 s clean narrator reference clip
- Runs locally on MPS; no GPU rental needed
- Already partially attempted — see [cumberbatch_xtts_embedded.wav](cumberbatch_xtts_embedded.wav)

**Path B — fine-tune (for max fidelity):**
- Re-segment dataset (VAD, 2–12 s) → re-transcribe (Whisper large-v3) → diarize and keep narrator → fine-tune **XTTS-v2** or **StyleTTS2** on a rented GPU → run inference locally on MPS
- Do **not** revive the Tacotron2/Coqui path

## Environment

```bash
source .venv/bin/activate     # Python 3.10
# requirements.txt is currently pinned for the deprecated Coqui/Tacotron2 setup —
# when moving to F5-TTS / XTTS-v2, install fresh into a new venv rather than reusing this one.
```

## Conventions

- Code runs **locally on the user's Mac**, not on a remote machine (this project is the exception to the global "I run the code remotely" rule).
- Prefer modifying existing notebooks over creating new ones; only create new notebooks when the existing flow doesn't fit.
- Do not commit `.env`, `dataset/chunks/`, `raw_cumberbatch_data/`, or `tts_output/` — these are large and/or user-private.
