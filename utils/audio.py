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
