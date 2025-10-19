from typing import Iterable, Optional

from metrics.clap.backends import HFProcessorBackend, LASSBackend, LaionBackend
from metrics.clap.backends.base import BackendUnavailable
from metrics.clap.types import CLAPItem, CLAPScored

_BACKENDS: dict[str, type] = {
    LaionBackend.name: LaionBackend,
    HFProcessorBackend.name: HFProcessorBackend,
    LASSBackend.name: LASSBackend,
}


def available_backends() -> list[str]:
    names: list[str] = []
    try:
        import laion_clap  # type: ignore
        names.append(LaionBackend.name)
    except Exception:
        pass
    try:
        from transformers import ClapProcessor, ClapModel  # noqa: F401
        names.append(HFProcessorBackend.name)
        names.append(LASSBackend.name)
    except Exception:
        pass
    seen: set[str] = set()
    uniq: list[str] = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def _make_backend(backend: str, device: Optional[str]):
    key = (backend or "").strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {', '.join(_BACKENDS.keys())}")
    try:
        return _BACKENDS[key](device=device or "auto")
    except BackendUnavailable as e:
        raise RuntimeError(f"Backend '{backend}' unavailable: {e}") from e


def calculate_scores(
    items: Iterable[CLAPItem],
    device: Optional[str] = None,
    *,
    backend: str = "laion_module",
) -> list[CLAPScored]:
    be = _make_backend(backend, device)
    return be.score_batch(items)
