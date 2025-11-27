from __future__ import annotations

from dataclasses import dataclass

TARGET_SR = 48_000


@dataclass(frozen=True)
class CLAPItem:
    id: str
    prompt: str
    audio_path: str


@dataclass(frozen=True)
class CLAPScored:
    item: CLAPItem
    clap_score: float
    audio_embedding: list[float] = None
    text_embedding: list[float] = None
