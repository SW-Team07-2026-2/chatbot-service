import logging
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings", "logger"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 필수
    LLM_MODEL: str
    ENVIRONMENT: str
    REDIS_URL: str
    CORS_ORIGINS: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        lower = v.strip().lower()
        if "dev" not in lower and "pro" not in lower:
            raise ValueError(
                f"유효하지 않은 ENVIRONMENT입니다: {v}. 'dev'(개발) 또는 'prod'(운영)를 포함해야 합니다."
            )
        return lower

    @property
    def is_production(self) -> bool:
        return "pro" in self.ENVIRONMENT


settings = Settings()
logger = logging.getLogger("uvicorn.error")