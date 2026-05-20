"""Shared helpers used by posthoc_eval.py.

Layout:
  audio_io.py   — WAV reading, mono conversion, resampling
  identity.py   — ECAPA-TDNN speaker embeddings, cosine sim
  inference.py  — MPS cache reset, audio segmentation utilities
  metrics.py    — WER + DNSMOS scoring
  transcribe.py — Whisper load + transcribe (16 kHz mono)
"""
