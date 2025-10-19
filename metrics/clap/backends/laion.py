from typing import Iterable
import numpy as np
import torch
import torchaudio

from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from metrics.clap.backends.base import BackendUnavailable, _resolve_device


class LaionBackend:
    name = "laion_module"

    def __init__(self, device: str) -> None:
        self.device = _resolve_device(device)
        try:
            from laion_clap import CLAP_Module  # type: ignore
        except Exception as e:
            raise BackendUnavailable("Install `laion-clap` to use laion_module backend.") from e
        self.model = CLAP_Module(enable_fusion=True)
        self.model.load_ckpt()
        self.model.eval().to(self.device)
        if not self.model.training and not self.model.training:  # double-check eval mode
            pass
        else:
            raise BackendUnavailable("CLAP_Module not in eval mode after load_ckpt().")

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        out: list[CLAPScored] = []
        for it in items:
            wav, sr = torchaudio.load(it.audio_path)
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            wav = wav.to(self.device)
            aemb = self.model.get_audio_embedding_from_data(wav, use_tensor=True)
            temb = self.model.get_text_embedding([it.description], use_tensor=True)
            aemb = torch.nn.functional.normalize(aemb, dim=-1)
            temb = torch.nn.functional.normalize(temb, dim=-1)
            val = torch.nn.functional.cosine_similarity(aemb, temb).item()
            out.append(CLAPScored(item=it, clap_score=float(np.round(val, 6))))
        return out
