from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, UUID4


class GenerateAudioRequest(BaseModel):
    """Client sends only a prompt."""
    prompt: Annotated[str, Field(min_length=1, max_length=800)]


class GenerateAudioResponse(BaseModel):
    """Server returns an id, the improved prompt, and a presigned URL to the audio."""
    audio_id: UUID4 = Field(default_factory=uuid4)
    improved_prompt: str
    audio_url: HttpUrl
