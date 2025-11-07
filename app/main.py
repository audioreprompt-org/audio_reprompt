from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.clients.s3 import S3Client
from app.routes import routes

from models.musicgen.model import load_musicgen_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Take the bucket and region for the config
    app.state.s3 = S3Client(bucket="audio_generations", region="us-west-2")
    app.state.synthesizer = load_musicgen_pipeline()

    yield


app = FastAPI(title='audio_prompt')


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(routes)
