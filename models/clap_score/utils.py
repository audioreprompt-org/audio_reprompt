import os
import random
import numpy as np
from typing import Optional

import torch
import torchaudio


def load_audio_tensor(audio_path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(dtype=torch.float32)
    return wav


def resolve_device(device: Optional[str] = "") -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device



def set_reproducibility(seed: int = 42):
    """
    Fija todas las semillas y configuraciones necesarias para que
    los resultados (embeddings, scores, etc.) sean reproducibles en CLAP o PyTorch.
    """
    print(f"Estableciendo modo determinista con semilla {seed}")

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modo determinista completo (puede afectar rendimiento, pero asegura reproducibilidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("Semillas y modo determinista configurados.\n")
