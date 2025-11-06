from fastapi import APIRouter


from app.routes.audio import app_audio
from app.routes.health import app_health

routes = APIRouter()

routes.include_router(app_audio, prefix='/api')
routes.include_router(app_health)
