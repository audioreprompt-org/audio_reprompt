from fastapi import APIRouter
from fastapi.responses import JSONResponse

app_health = APIRouter(prefix='/health')


@app_health.get(
    path='/',
    tags=['Health']
)
async def health_check():
    return JSONResponse(status_code=200, content={'success': True})
