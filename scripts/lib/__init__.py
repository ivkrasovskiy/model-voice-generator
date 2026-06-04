"""Shared helpers used by posthoc_eval.py and accent_coach scripts.

Layout:
  audio_io.py   — WAV reading, mono conversion, resampling
  identity.py   — ECAPA-TDNN speaker embeddings, cosine sim
  inference.py  — MPS cache reset, audio segmentation utilities
  manifest.py   — JSON manifest loading and WAV path resolution
  metrics.py    — WER + DNSMOS scoring
  transcribe.py — Whisper load + transcribe (16 kHz mono)
"""
