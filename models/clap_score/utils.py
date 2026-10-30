import os
import random
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio_tensor(
    audio_path: Union[str, os.PathLike], sample_rate: int
) -> torch.Tensor:
    """
    Loads an audio file into a torch.Tensor of shape (1, T) and resamples to sample_rate.
    Uses soundfile as primary loader (with librosa and torchaudio fallbacks) so it does
    not require torchcodec (which torchaudio >= 2.9.0 defaults to on Linux).
    """
    audio_path_str = os.fspath(audio_path)
    try:
        data, sr = sf.read(audio_path_str, dtype="float32", always_2d=True)
        # sf.read returns (frames, channels) -> transpose to (channels, frames)
        wav = torch.from_numpy(data.T).contiguous()
    except Exception:
        try:
            import librosa

            data, sr = librosa.load(audio_path_str, sr=None, mono=False)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            wav = torch.from_numpy(data.astype(np.float32, copy=False)).contiguous()
        except Exception:
            wav, sr = torchaudio.load(audio_path_str)

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
