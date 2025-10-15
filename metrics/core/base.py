from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


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
    - Return a structured Dict for the engine to aggregate
    """

    name: str = "metric"

    def run(
        self,
        prompts: List[PromptRow],
        audio_items: List[AudioItem],
        metric_cfg: Dict[str, Any],
        device: str,
        scores_dir: Path
    ) -> Dict[str, Any]:
        raise NotImplementedError
