from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

__all__ = ["setting", "logger", "constant"]

# env
class Settings(BaseSettings):
    
    #=====================================================

    # 서버
    ENVIRONMENT: str
    REDIS_URL: str
    CORS_ORIGINS: str
    BACKEND_URL: str

    # AI
    LLM_MODEL: str
    GEMINI_API_KEY: str

    #TAVILY_API_KEY: str # ai 웹 검색용
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
    LEGACY_COOKIE_NAME: str = "access_token"  # 목서버 및 하위 호환 쿠키 명
    AUTH_HTTP_TIMEOUT_SECONDS: float = 3.0  # jwt 검증 대기 시간

constant = Constants()

# 로거
logger = logging.getLogger("uvicorn.error")
