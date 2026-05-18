"""Shared helpers used across data-processing, training, and eval scripts.

Each module is small and single-purpose. Import only what you need.

Layout:
  audio_io.py   — WAV reading, mono conversion, resampling
  dataset.py    — metadata.csv read/write, deterministic train/val splits
  identity.py   — ECAPA-TDNN speaker embeddings, cosine sim, centroids
  inference.py  — F5-TTS wrapper, multi-segment generation with retry
  transcribe.py — Whisper load + transcribe (16 kHz mono)
"""
