from datetime import datetime, timedelta, timezone
import logging

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["setting", "logger", "constant", "KST", "to_kst"]

KST = timezone(timedelta(hours=9))


def to_kst(ts: float | None = None) -> datetime:
    """epoch timestamp를 한국 표준시(KST) datetime으로 변환"""
    return datetime.fromtimestamp(ts if ts is not None else datetime.now(timezone.utc).timestamp(), tz=timezone.utc).astimezone(KST)


# env
class Settings(BaseSettings):

    #=====================================================

    # 서버
    ENVIRONMENT: str
    REDIS_URL: str
    CORS_ORIGINS: str
    BACKEND_URL: str
    BACKEND_HOST: str = ""

    # AI
    LLM_MODEL: str
    GEMINI_API_KEY: str

    EXA_API_KEY: str # ai 웹 검색용

    #=====================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        lower = v.strip().lower()
        if "dev" not in lower and "pro" not in lower:
            raise ValueError(
                f"유효하지 않은 ENVIRONMENT입니다: {v}. 'dev'(개발) 또는 'prod'(운영)를 포함해야 합니다."
            ) from None
        return lower

    @property
    def is_production(self) -> bool:
        return "pro" in self.ENVIRONMENT


setting = Settings()


# 상수
class Constants:
    # 인증 (실제 백엔드 Gateway G3 명세서 기준: GET /api/auth/me)
    AUTH_PATH: str = "/api/auth/me"
    AUTH_URL: str = setting.BACKEND_URL.rstrip("/") + AUTH_PATH
    AUTH_COOKIE_NAME: str = "AT"  # 실제 백엔드 액세스 토큰 쿠키 명 (HttpOnly)
    LEGACY_COOKIE_NAME: str = "access_token"  # 하위 호환 쿠키 명
    AUTH_HTTP_TIMEOUT_SECONDS: float = 3.0  # jwt 검증 대기 시간

    # 장기 기억 (백엔드 Gateway M26/M27 & Redis LTM)
    LTM_HISTORY_PATH: str = "/api/members/me/chat-history"
    LTM_HISTORY_URL: str = setting.BACKEND_URL.rstrip("/") + LTM_HISTORY_PATH
    LTM_MAX: int = 50
    LTM_TTL: int = 60 * 60 * 6
    LTM_MAX_CONTENT_LENGTH: int = 200
    LTM_LOCK_TTL_SECONDS: int = 5
    BACKEND_HTTP_TIMEOUT_SECONDS: float = 5.0

    # 단기 대화 & 에이전트
    CHAT_MAX: int = 30  # 최신 메시지 유지 갯수
    CHAT_TTL: int = 60 * 60 * 6  # TTL (초)
    AGENT_REQUEST_LIMIT: int = 6  # 요청당 모델 최대 호출 횟수
    TOOL_TIMEOUT_SECONDS: float = 30.0  # 도구 대기 시간 (초)
    TOOL_HISTORY_MAX_CHARS: int = 300  # 히스토리 저장 시 도구 결과 최대 길이
    LLM_TIMEOUT_MS: int = 60_000  # 모델 호출 타임아웃 (밀리초)
    LLM_RETRY_ATTEMPTS: int = 3  # 모델 호출 시도 횟수 (최초 1회 + 재시도 2회)

    # 파일 업로드 & 멀티모달
    MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024
    ALLOWED_GEMINI_FILE_URI_PREFIX: str = "https://generativelanguage.googleapis.com/"


constant = Constants()

# 로거
logger = logging.getLogger("uvicorn.error")
