from fastapi import APIRouter, HTTPException

from app.models.audio import GenerateAudioRequest, GenerateAudioResponse
from pydantic import HttpUrl


app_audio = APIRouter(prefix='/audio')


@app_audio.post(
    path='/generate',
    status_code=200,
    tags=['Audio'],
    response_model=GenerateAudioResponse,
    summary="...",
)
async def generate_audio(payload: GenerateAudioRequest) -> GenerateAudioResponse:
    """
    Perform the application get of all vehicle
    """
    try:
        improved = payload.prompt.strip().capitalize()
        url = HttpUrl("https://cdn.example.com/audio")

        return GenerateAudioResponse(
            improved_prompt=improved,
            audio_url=url,
        )
    except Exception as error:
        message = 'Error occurred generating audio'
        logger.error(f'{message}| Error: {repr(error)}')
        raise HTTPException(status_code=500, detail="Error occurred generating audio")
