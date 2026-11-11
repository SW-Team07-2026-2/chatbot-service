from fastapi import APIRouter, Response, status

from app.server.redis import ping

router = APIRouter(tags=["health"])

@router.get("/health/liveness")
async def liveness():
    return {"status": "UP"}


@router.get("/health/readiness")
async def readiness(res: Response):
    redis_up = await ping()
    if not redis_up:
        res.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "DOWN", "components": {"redis": "DOWN"}}
    return {"status": "UP", "components": {"redis": "UP"}}