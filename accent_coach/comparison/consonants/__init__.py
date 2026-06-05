"""Consonant quality comparison subpackage.

Public API: score_consonants(user, audio, sr, ...) -> ConsonantScore
"""
from .aggregator import score_consonants

__all__ = ["score_consonants"]
