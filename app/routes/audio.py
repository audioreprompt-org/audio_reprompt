import logging

from fastapi import APIRouter, HTTPException, Request
from uuid import uuid4

from app.models.audio import GenerateAudioRequest, GenerateAudioResponse
from pydantic import HttpUrl

from models.musicgen.audio_generator import generate_audio_buffer_from_prompt

app_audio = APIRouter(prefix='/audio')
logger = logging.getLogger(__name__)


@app_audio.post(
    path='/generate',
    status_code=200,
    tags=['Audio'],
    response_model=GenerateAudioResponse,
    summary="...",
)
async def generate_audio(payload: GenerateAudioRequest, request: Request) -> GenerateAudioResponse:
    """
    Perform the application get of all vehicle
    """
    try:
        # Load needed clients
        s3 = request.app.state.s3
        musicgen = request.app.state.synthesizer

        # Improve the prompt
        improved = payload.prompt.strip().capitalize()

        # Generate the audio with the prompt
        audio_buffer = generate_audio_buffer_from_prompt(improved, synthesiser=musicgen)

        # Save the audio buffer
        audio_id = uuid4()
        presigned_url = s3.upload_object(data=audio_buffer, key=str(audio_id), expires_in=600)

        return GenerateAudioResponse(
            audio_id=audio_id,
            improved_prompt=improved,
            audio_url=HttpUrl(presigned_url),
        )
    except Exception as error:
        message = 'Error occurred generating audio'
        logger.error(f'{message}| Error: {repr(error)}')
        raise HTTPException(status_code=500, detail="Error occurred generating audio")
