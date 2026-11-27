from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import setting, logger
from app.server import http, redis
from app.server.api import router as server_router
from app.chatbot import skill
from app.chatbot.api import router as chatbot_router
from app.recommender import router as recommender_router, service as recommender_service


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

    await recommender_service.startup()

    yield

    # 서버 종료 시 백그라운드 작업 완료 대기 및 커넥션 풀 정리
    await recommender_service.shutdown()
    await skill.drain_background_tasks()
    await http.close()
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

app.include_router(server_router)
app.include_router(chatbot_router)
app.include_router(recommender_router)
