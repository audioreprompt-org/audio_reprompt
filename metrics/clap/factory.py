from typing import Iterable, Optional, List

import torch

from metrics.clap.backends import HFProcessorBackend, LASSBackend, LaionBackend
from metrics.clap.backends.base import BackendUnavailable
from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from utils.audio import load_audio_tensor
from utils.device import resolve_device

_BACKENDS: dict[str, type] = {
    LaionBackend.name: LaionBackend,
    HFProcessorBackend.name: HFProcessorBackend,
    LASSBackend.name: LASSBackend,
}


def available_backends() -> list[str]:
    names: list[str] = []
    try:
        import laion_clap
        names.append(LaionBackend.name)
    except Exception:
        pass

    try:
        from transformers import ClapProcessor, ClapModel
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


def get_audio_embeddings_from_paths(
    audio_paths: list[str],
    device: Optional[str] = None,
    *,
    backend: str = "laion_module",
) -> List[torch.Tensor]:
    device = resolve_device(device)
    model = _make_backend(backend, device)
    embeddings = []
    for audio_path in audio_paths:
        audio_tensor = load_audio_tensor(audio_path, TARGET_SR).to(device)
        emb = model.embed_audio(audio_tensor)

        embeddings.append(emb)
    return embeddings


def get_text_embeddings(
    texts: list[str],
    device: Optional[str] = None,
    *,
    backend: str = "laion_module",
) -> List[torch.Tensor]:
    device = resolve_device(device)
    model = _make_backend(backend, device)
    embeddings = []
    for text in texts:
        emb = model.embed_text(text)

        embeddings.append(emb)
    return embeddings
