from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import logger, settings
from app.server.__memory import check_redis_connection, close_redis_connection
from app.api.chat import router as chat_router
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"실행 환경: {'운영(production) - /docs 비활성화' if settings.is_production else '개발(development) - /docs 활성화'}"
    )

    # 서버 시작 시 Redis 연결 엄격 검증
    # 연결 실패 시 장황한 프레임워크 트레이스백 없이 깔끔하게 실패 종료
    orig_tracebacklimit = getattr(sys, "tracebacklimit", 1000)
    sys.tracebacklimit = 0
    # main.py lifespan 수정 예시
try:
    await check_redis_connection()
except Exception as e:
    logger.warning(f"⚠️ Redis가 꺼져 있지만 서버를 계속 시작합니다. 나중에 켜지면 자동 연결됩니다: {e}")

    yield
    # 서버 종료 시 커넥션 풀 정리
    await close_redis_connection()


app = FastAPI(
    title="INDIG AI Chat Service",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(health_router)
