from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import routes

app = FastAPI(title="audio_prompt")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes)
