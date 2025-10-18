"""
Top-level metrics package.

Public API:
- CLAP scoring: CLAPItem, CLAPScored, clap_score()
- Core types: Metric, PromptRow, AudioItem
(Other metrics are intentionally unimplemented right now.)
"""

from metrics.clap import CLAPItem, CLAPScored, calculate_scores
from metrics.core import AudioItem, PromptRow, Metric

__all__ = [
    "CLAPItem", "CLAPScored", "calculate_scores",
    "Metric", "PromptRow", "AudioItem",
]
