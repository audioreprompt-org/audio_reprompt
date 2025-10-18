from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRow:
    id: str
    instrument: str
    description: str


@dataclass(frozen=True)
class AudioItem:
    id: str
    instrument: str
    description: str
    audio_path: str


class Metric:
    """Pluggable metric interface. Implementations should be side-effect-light:
    - Read inputs from arguments
    - Write their own outputs (CSV/JSON) when appropriate
    - Return a structured dict for the engine to aggregate
    """
    name: str = "metric"

    def run(
            self,
            prompts: list[PromptRow],
            audio_items: list[AudioItem],
            metric_cfg: dict[str, Any],
            device: str,
            scores_dir: Path
    ) -> dict[str, Any]:
        raise NotImplementedError
