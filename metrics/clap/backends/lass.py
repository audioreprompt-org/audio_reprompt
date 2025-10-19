from typing import Iterable
import numpy as np
import torch
import torchaudio

from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from metrics.clap.backends.base import BackendUnavailable, _resolve_device


_EXPECTED_MODEL_ID = "laion/clap-htsat-fused"


class LASSBackend:
    name = "lass"

    def __init__(self, device: str) -> None:
        self.device = _resolve_device(device)
        try:
            from transformers import ClapProcessor, ClapModel  # type: ignore
        except Exception as e:
            raise BackendUnavailable("Install `transformers` to use lass backend.") from e
        try:
            self.processor = ClapProcessor.from_pretrained(_EXPECTED_MODEL_ID)
            self.model = ClapModel.from_pretrained(_EXPECTED_MODEL_ID)
            self.model.eval().to(self.device)
        except Exception as e:
            raise BackendUnavailable(f"Failed to load HF model '{_EXPECTED_MODEL_ID}' for LASS backend.") from e

        name_or_path = getattr(self.model, "name_or_path", None) or getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if name_or_path and _EXPECTED_MODEL_ID not in str(name_or_path):
            raise BackendUnavailable(f"Unexpected HF model source: '{name_or_path}' (expected '{_EXPECTED_MODEL_ID}')")

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        out: list[CLAPScored] = []
        for it in items:
            wav, sr = torchaudio.load(it.audio_path)
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            audio_np = wav.squeeze(0).cpu().numpy()

            audio_inputs = self.processor(
                audios=[audio_np],
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = self.processor(
                text=[it.description],
                return_tensors="pt",
                padding=True,
            )
            audio_inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in audio_inputs.items()}
            text_inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in text_inputs.items()}

            aemb = self.model.get_audio_features(**audio_inputs)
            temb = self.model.get_text_features(**text_inputs)
            aemb = torch.nn.functional.normalize(aemb, dim=-1)
            temb = torch.nn.functional.normalize(temb, dim=-1)
            val = torch.nn.functional.cosine_similarity(aemb, temb).item()
            out.append(CLAPScored(item=it, clap_score=float(np.round(val, 6))))
        return out
