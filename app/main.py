from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router, health_router
from app.config import setting, logger
from app.server import redis, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"{'운영모드 - /docs, /redoc 비활성화' if setting.is_production else '개발모드 - /docs, /redoc 활성화'}"
    )

    # 서버 시작 시 Redis 연결 상태 점검
    redis_up = await redis.ping()
    if redis_up:
        logger.info(f"Redis 연결 완료 ({setting.REDIS_URL})")
    else:
        logger.warning(
            f"Redis 연결 실패 ({setting.REDIS_URL})"
        )

    yield

    # 서버 종료 시 커넥션 풀 정리
    await auth.close()
    await redis.close()
    logger.info("Redis 연결 풀 정리 완료")


app = FastAPI(
    title="INDIG AI Chat Service",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if setting.is_production else "/docs",
    redoc_url=None if setting.is_production else "/redoc",
    openapi_url=None if setting.is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=setting.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(health_router)
