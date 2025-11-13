from contextlib import asynccontextmanager

from typing import cast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import State

from clients.s3 import S3Client
from routes import routes
from config import get_environment_config

from models.musicgen.model import load_musicgen_pipeline


ENV_CFG = get_environment_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = cast(State, app.state)
    # TODO: Take the bucket and region for the config
    state.s3 = S3Client(bucket=ENV_CFG.bucket_name, region=ENV_CFG.aws_region)
    # state.synthesizer = load_musicgen_pipeline()
    yield


app = FastAPI(title="audio_prompt", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes)
