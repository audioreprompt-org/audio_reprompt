import os
from functools import lru_cache

from openai import OpenAI

from models.music_curator.prompts import MCU_PROMPTS

KIMI_K2_THINKING_MODEL = "kimi-k2-thinking-turbo"
OPENAI_GPT_5_NANO_MODEL = "gpt-5-nano"


MUSIC_CURATOR_ROLE = """
You are MCU, an AI music curator assistant that provide recommendations using musician vocabulary.
"""


@lru_cache(maxsize=2)
def get_client(model: str):
    return (
        OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if model == OPENAI_GPT_5_NANO_MODEL
        else OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.ai/v1",
        )
    )


def mcu_reprompt(
    crossmodal_descriptors: str,
    music_captions: str,
    model: str = KIMI_K2_THINKING_MODEL,
    prompt_version: str = "V3",
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    messages = [
        {"role": "system", "content": MUSIC_CURATOR_ROLE},
        {
            "role": "user",
            "content": MCU_PROMPTS[prompt_version].format(
                crossmodal_descriptors=crossmodal_descriptors,
                music_captions=music_captions,
            ),
        },
    ]

    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None and "thinking" not in model:
        kwargs["temperature"] = temperature
    if top_p is not None and "thinking" not in model:
        kwargs["top_p"] = top_p

    response = get_client(model).chat.completions.create(**kwargs)

    return response.choices[0].message.content
