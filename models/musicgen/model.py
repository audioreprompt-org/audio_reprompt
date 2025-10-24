import logging
from typing import Optional

import torch
from transformers import AutoModelForTextToAudio, AutoProcessor, pipeline
from transformers.pipelines import Pipeline

logger = logging.getLogger(__name__)


def load_musicgen_pipeline(
    model_name: str = "csc-unipd/tasty-musicgen-small",
    prefer_gpu: bool = True,
    low_cpu_mem_usage: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    device = "cuda:0" if prefer_gpu and torch.cuda.is_available() else "cpu"
    if dtype is None and device.startswith("cuda"):
        dtype = torch.float16

    model = AutoModelForTextToAudio.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if "cuda" in device else None,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    if "cuda" in device:
        _ = model.to(device)
        torch.cuda.synchronize()

    pipe: Pipeline = pipeline(
        task="text-to-audio",
        model=model,
        tokenizer=processor,
    )
    return pipe


if __name__ == "__main__":
    musicgen = load_musicgen_pipeline()
    output = musicgen(
        "classical style ambient melody with piano and strings",
        forward_params={"do_sample": True},
    )
    print(f"Generated sample rate: {output['sampling_rate']}")
