from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class AestheticsItem:
    id: str
    audio_path: str
    instrument: Optional[str] = None


@dataclass
class AestheticsScored(AestheticsItem):
    aesthetics: float = float("nan")


class _UnavailableAesthetics(Exception):
    """Raised when the Audiobox-Aesthetics metric is requested but not configured."""
    pass


def score(
        items: Iterable[AestheticsItem],
        device: Optional[str] = None,
        clamp_seconds: Optional[float] = None
) -> list[AestheticsScored]:
    """
    Not implemented on purpose in this phase. We expose the function with the correct
    signature so call sites fail fast with a helpful message if invoked.
    """
    raise _UnavailableAesthetics(
        "Aesthetics metric is not implemented in this phase (CLAP-only). "
        "Enable later by providing a model/weights and implementing metrics.aesthetics.score()."
    )
