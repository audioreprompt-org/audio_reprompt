import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import routes

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
app = FastAPI(title="audio_prompt")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes)
