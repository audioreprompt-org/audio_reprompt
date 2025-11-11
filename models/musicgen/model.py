import logging
from typing import Optional

import torch
from transformers import pipeline
from transformers.pipelines import Pipeline

logger = logging.getLogger(__name__)


def load_musicgen_pipeline(
    model_name: str = "csc-unipd/tasty-musicgen-small",
    prefer_gpu: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    if dtype is None and device.startswith("cuda"):
        dtype = torch.float16

    pipe: Pipeline = pipeline(
        task="text-to-audio", model=model_name, torch_dtype=dtype, device=device
    )

    logger.info(f"Modelo MusicGen cargado correctamente en {device}")

    return pipe
