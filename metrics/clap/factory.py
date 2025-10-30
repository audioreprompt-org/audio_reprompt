from typing import Iterable, Optional, Sequence, Union

import torch

from metrics.clap.backends import HFProcessorBackend, LASSBackend, LaionBackend
from metrics.clap.backends.base import BackendUnavailable
from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from utils.audio import load_audio_tensor
from utils.device import resolve_device


TensorLike = Union[torch.Tensor, Sequence[torch.Tensor]]

_BACKENDS: dict[str, type] = {
    LaionBackend.name: LaionBackend,
    HFProcessorBackend.name: HFProcessorBackend,
    LASSBackend.name: LASSBackend,
}

_BACKEND_SINGLETON = None
_BACKEND_CURRENT_KEY: Optional[str] = None


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


def _normalize_key(backend: Optional[str]) -> str:
    return (backend or "").strip().lower()


def _make_backend(backend: str, device: Optional[str], backend_cfg: Optional[dict] = None):
    key = _normalize_key(backend)
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {', '.join(_BACKENDS.keys())}")
    try:
        # Pass config only to backends that accept it (LaionBackend does).
        if key == LaionBackend.name:
            return _BACKENDS[key](device=device or "auto", **(backend_cfg or {}))
        return _BACKENDS[key](device=device or "auto")
    except BackendUnavailable as e:
        raise RuntimeError(f"Backend '{backend}' unavailable: {e}") from e



def _get_backend(backend: str, device: Optional[str], backend_cfg: Optional[dict] = None):
    """
    Return the singleton backend. Re-create when backend name OR backend_cfg changes.
    Device changes are ignored after the first initialization to keep it simple.
    """
    import json

    global _BACKEND_SINGLETON, _BACKEND_CURRENT_KEY

    key = _normalize_key(backend)
    cfg_sig = json.dumps(backend_cfg or {}, sort_keys=True, ensure_ascii=False)
    cache_key = f"{key}|cfg:{cfg_sig}"
    if (_BACKEND_SINGLETON is None) or (_BACKEND_CURRENT_KEY != cache_key):
        _BACKEND_SINGLETON = _make_backend(backend, device, backend_cfg=backend_cfg)
        _BACKEND_CURRENT_KEY = cache_key
    return _BACKEND_SINGLETON


def calculate_scores(
    items: Iterable[CLAPItem],
    device: Optional[str] = None,
    *,
    backend: str = "laion_module",
    backend_cfg: Optional[dict] = None,
) -> list[CLAPScored]:
    be = _get_backend(backend, device, backend_cfg=backend_cfg)
    return be.score_batch(items)


def calculate_scores_with_embeddings(
    audio_embeddings: TensorLike,
    text_embeddings: TensorLike,
) -> list[float]:
    """
    Calcula similitudes coseno entre pares de embeddings de audio y texto.
    Acepta listas de tensores (len N, cada uno (D,)) o tensores apilados (N, D).

    Retorna una lista de floats (N,) con los CLAP scores por par.
    """
    if isinstance(audio_embeddings, torch.Tensor):
        a = audio_embeddings
    else:
        a = torch.stack(list(audio_embeddings), dim=0)

    if isinstance(text_embeddings, torch.Tensor):
        t = text_embeddings
    else:
        t = torch.stack(list(text_embeddings), dim=0)

    if a.shape != t.shape:
        raise ValueError(f"Shape mismatch: audio {a.shape} vs text {t.shape} (esperado N x D emparejado)")

    if a.device != t.device:
        t = t.to(a.device)

    a = torch.nn.functional.normalize(a, dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    sims = (a * t).sum(dim=-1)

    return sims.detach().cpu().tolist()


def get_audio_embeddings_from_paths(
    audio_paths: list[str],
    device: Optional[str] = None,
    *,
    backend: str = "laion_module",
    backend_cfg: Optional[dict] = None,
) -> list[torch.Tensor]:
    device = resolve_device(device)
    model = _get_backend(backend, device, backend_cfg=backend_cfg)
    embeddings: list[torch.Tensor] = []
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
    backend_cfg: Optional[dict] = None,
) -> list[torch.Tensor]:
    device = resolve_device(device)
    model = _get_backend(backend, device, backend_cfg=backend_cfg)
    embeddings: list[torch.Tensor] = []
    for text in texts:
        emb = model.embed_text(text)
        embeddings.append(emb)
    return embeddings
