from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TARGET_SR = 48_000


@dataclass(frozen=True)
class CLAPItem:
    id: str
    description: str
    audio_path: str
    instrument: Optional[str] = None


@dataclass(frozen=True)
class CLAPScored:
    item: CLAPItem
    clap_score: float
