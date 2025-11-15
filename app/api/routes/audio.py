import logging

from fastapi import APIRouter, HTTPException, Request
from uuid import uuid4

from model.audio import GenerateAudioRequest, GenerateAudioResponse

from test.audio import generate_local_fake_audio, TMP_CACHE_DIR

from clients.runpod import call_runpod_musicgen


app_audio = APIRouter(prefix="/audio")
logger = logging.getLogger(__name__)

MOCKED_RESPONSE = True


@app_audio.post(
    path="/generate",
    status_code=200,
    tags=["Audio"],
    response_model=GenerateAudioResponse,
    summary="Perform the application get a piece of audio given the modified prompt.",
)
async def generate_audio(
    payload: GenerateAudioRequest, request: Request
) -> GenerateAudioResponse:
    try:
        # Generate audio identifier.
        audio_id = uuid4()

        # Improve the prompt
        improved = payload.prompt.strip().capitalize()

        if MOCKED_RESPONSE:
            audio_base64 = generate_local_fake_audio(audio_id)

            return GenerateAudioResponse(
                audio_id=audio_id,
                improved_prompt=improved,
                audio_base64=audio_base64,
            )

        # Generate the audio with the prompt
        audio_base64 = call_runpod_musicgen(improved)

        return GenerateAudioResponse(
            audio_id=audio_id,
            improved_prompt=improved,
            audio_base64=audio_base64,
        )
    except Exception as error:
        message = "Error occurred generating audio"
        logger.error(f"{message}| Error: {repr(error)}")
        raise HTTPException(status_code=500, detail="Error occurred generating audio")
