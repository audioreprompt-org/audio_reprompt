import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from models.clap_score.model import ClapModel
from models.clap_score.utils import load_audio_tensor


@pytest.fixture
def mono_wav(tmp_path):
    sr = 32000
    duration_s = 0.5
    t = np.linspace(
        0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32
    )
    # 440 Hz sine wave
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "mono.wav"
    sf.write(str(wav_path), data, sr, subtype="FLOAT")
    return wav_path, sr, data


@pytest.fixture
def stereo_wav(tmp_path):
    sr = 32000
    duration_s = 0.5
    t = np.linspace(
        0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32
    )
    ch1 = 0.4 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.2 * np.sin(2 * np.pi * 880 * t)
    data = np.stack([ch1, ch2], axis=-1)  # (frames, 2)
    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), data, sr, subtype="FLOAT")
    return wav_path, sr, data


def test_load_audio_tensor_shape_and_dtype(mono_wav):
    wav_path, sr, orig_data = mono_wav
    tensor = load_audio_tensor(wav_path, sample_rate=sr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == len(orig_data)
    assert tensor.dtype == torch.float32
    # Verify values match original
    np.testing.assert_allclose(tensor.squeeze(0).numpy(), orig_data, atol=1e-5)


def test_load_audio_tensor_stereo_downmix(stereo_wav):
    wav_path, sr, orig_data = stereo_wav
    tensor = load_audio_tensor(wav_path, sample_rate=sr)

    assert tensor.ndim == 2
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == orig_data.shape[0]
    # Downmix should equal mean across channels
    expected = orig_data.mean(axis=1)
    np.testing.assert_allclose(tensor.squeeze(0).numpy(), expected, atol=1e-5)


def test_load_audio_tensor_resampling(mono_wav):
    wav_path, orig_sr, orig_data = mono_wav
    target_sr = 48000
    tensor = load_audio_tensor(wav_path, sample_rate=target_sr)

    expected_len = int(len(orig_data) * target_sr / orig_sr)
    assert tensor.shape[0] == 1
    assert abs(tensor.shape[1] - expected_len) <= 1
    assert tensor.dtype == torch.float32


def test_load_audio_tensor_without_torchcodec(mono_wav, monkeypatch):
    """
    Simulates Linux environment where torchaudio >= 2.9 raises TorchCodec ImportError.
    load_audio_tensor should cleanly load audio via soundfile without calling torchaudio.load.
    """
    wav_path, sr, _ = mono_wav

    def mock_torchaudio_load(*args, **kwargs):
        raise ImportError(
            "TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function."
        )

    monkeypatch.setattr(torchaudio, "load", mock_torchaudio_load)

    tensor = load_audio_tensor(wav_path, sample_rate=48000)
    assert tensor.shape[0] == 1
    assert tensor.dtype == torch.float32


def test_load_audio_tensor_fallback_to_librosa(mono_wav, monkeypatch):
    """
    Verifies that if soundfile fails, load_audio_tensor falls back to librosa.
    """
    wav_path, sr, _ = mono_wav

    def mock_sf_read(*args, **kwargs):
        raise RuntimeError("Soundfile error")

    def mock_torchaudio_load(*args, **kwargs):
        raise ImportError(
            "TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function."
        )

    monkeypatch.setattr(sf, "read", mock_sf_read)
    monkeypatch.setattr(torchaudio, "load", mock_torchaudio_load)

    tensor = load_audio_tensor(wav_path, sample_rate=48000)
    assert tensor.shape[0] == 1
    assert tensor.dtype == torch.float32


def test_calculate_score_with_embeddings_scalar_and_batch():
    """
    Ensure calculate_score_with_embeddings returns list[float] for both N=1 and N>1.
    """
    # Single item (N=1)
    text_1 = torch.randn(1, 512)
    audio_1 = torch.randn(1, 512)
    scores_1 = ClapModel.calculate_score_with_embeddings(text_1, audio_1)
    assert isinstance(scores_1, list)
    assert len(scores_1) == 1
    assert isinstance(scores_1[0], float)

    # Batch (N=3)
    text_3 = torch.randn(3, 512)
    audio_3 = torch.randn(3, 512)
    scores_3 = ClapModel.calculate_score_with_embeddings(text_3, audio_3)
    assert isinstance(scores_3, list)
    assert len(scores_3) == 3
    assert all(isinstance(s, float) for s in scores_3)
