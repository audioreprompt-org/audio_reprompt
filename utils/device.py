from typing import Optional
import torch


def resolve_device(device: Optional[str]) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
