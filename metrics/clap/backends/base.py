from typing import Iterable, Any, Optional, Protocol
import torch

from metrics.clap.types import CLAPItem, CLAPScored


class BackendUnavailable(RuntimeError):
    pass


def _resolve_device(device: Optional[str]) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class BaseBackend(Protocol):
    name: str

    def __init__(self, device: str, backend_cfg: Optional[dict[str, Any]] = None) -> None: ...

    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]: ...
