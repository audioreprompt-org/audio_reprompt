import logging

from fastapi import APIRouter, HTTPException, Request
from uuid import uuid4, UUID

from starlette.responses import FileResponse

from model.audio import GenerateAudioRequest, GenerateAudioResponse
from pydantic import HttpUrl

from test.audio import generate_local_fake_audio, TMP_CACHE_DIR
from models.musicgen.audio_generator import generate_audio_buffer_from_prompt

app_audio = APIRouter(prefix="/audio")
logger = logging.getLogger(__name__)

MOCKED_RESPONSE = True


@app_audio.get("/download/{audio_id}")
async def download_audio(audio_id: UUID, request: Request) -> FileResponse:
    p = TMP_CACHE_DIR / f"{audio_id}.wav"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Audio not found or expired.")

    return FileResponse(
        path=str(p),
        media_type="audio/wav",
        filename=f"{audio_id}.wav",
        headers={"Cache-Control": "no-store"},
    )


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
        # Load needed clients.
        s3 = request.app.state.s3
        musicgen = request.app.state.synthesizer
        # Generate audio identifier.
        audio_id = uuid4()

        # Improve the prompt
        improved = payload.prompt.strip().capitalize()

        if MOCKED_RESPONSE:
            generate_local_fake_audio(audio_id)

            download_url = str(
                request.url_for("download_audio", audio_id=str(audio_id))
            )

            return GenerateAudioResponse(
                audio_id=audio_id,
                improved_prompt=improved,
                audio_url=download_url,
            )

        # Generate the audio with the prompt
        audio_buffer = generate_audio_buffer_from_prompt(improved, synthesiser=musicgen)

        # Save the audio buffer
        presigned_url = s3.upload_object(
            data=audio_buffer,
            key=str(audio_id) + ".wav",
            content_type="audio/wav",
            expires_in=600,
        )

        return GenerateAudioResponse(
            audio_id=audio_id,
            improved_prompt=improved,
            audio_url=HttpUrl(presigned_url),
        )
    except Exception as error:
        message = "Error occurred generating audio"
        logger.error(f"{message}| Error: {repr(error)}")
        raise HTTPException(status_code=500, detail="Error occurred generating audio")
