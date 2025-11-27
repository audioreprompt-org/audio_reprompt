from typing import TypedDict


class PromptItemLoader(TypedDict):
    id: str
    prompt: str


class OutputGeneratedAudioItem(TypedDict):
    id: str
    prompt: str
    audio_path: str
