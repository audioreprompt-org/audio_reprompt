"""
Top-level metrics package.

Public API:
- CLAP scoring: CLAPItem, CLAPScored, clap_score()
- Core types: Metric, PromptRow, AudioItem
(Other metrics are intentionally unimplemented right now.)
"""
from .clap import CLAPItem, CLAPScored, calculate_scores
from .core.base import Metric, PromptRow, AudioItem

__all__ = [
    "CLAPItem", "CLAPScored", "calculate_scores",
    "Metric", "PromptRow", "AudioItem",
]
