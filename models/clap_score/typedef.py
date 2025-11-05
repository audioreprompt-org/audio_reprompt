from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

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
audio_embedding: list[float] | None = None
text_embedding: list[float] | None = None
