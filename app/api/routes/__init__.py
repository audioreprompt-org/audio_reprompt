from fastapi import APIRouter
from .audio import app_audio
from .health import app_health

routes = APIRouter(prefix="/api", tags=["API"])

routes.include_router(app_audio)
routes.include_router(app_health)
