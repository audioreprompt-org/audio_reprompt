from functools import lru_cache
from openai import OpenAI
from .prompts import MCU_PROMPTS
from .config import get_moonshot_api_key

KIMI_K2_THINKING_MODEL = "kimi-k2-thinking"
KIMI_K2_THINKING_MODEL_TURBO = "kimi-k2-thinking-turbo"
MUSIC_CURATOR_ROLE = "You are MCU, an AI music curator assistant that provide recommendations using musician vocabulary."


@lru_cache(maxsize=2)
def get_client():
    return OpenAI(api_key=get_moonshot_api_key(), base_url="https://api.moonshot.ai/v1")


def mcu_reprompt(crossmodal_descriptors: str, music_captions: str, prompt_version: str = "V3") -> str:
    messages = [
        {"role": "system", "content": MUSIC_CURATOR_ROLE},
        {"role": "user", "content": MCU_PROMPTS[prompt_version].format(
            crossmodal_descriptors=crossmodal_descriptors,
            music_captions=music_captions
        )},
    ]
    response = get_client().chat.completions.create(model=KIMI_K2_THINKING_MODEL_TURBO, messages=messages)
    return response.choices[0].message.content
