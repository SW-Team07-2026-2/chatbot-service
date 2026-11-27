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

    # Recommender (선택 · 백엔드 /ingest/members/** 서명 키 및 관리자 API 시크릿)
    INGEST_MEMBER_KEY_ID: str = ""
    INGEST_MEMBER_SECRET: str = ""
    RECOMMENDER_ADMIN_SECRET: str = ""
    REC_ARTIFACT_DIR: str = "/tmp/indig_recommender"

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

    # 단기 대화 & 에이전트 (운영 Gateway 30초 제한 고려)
    CHAT_MAX: int = 40  # 최신 메시지 유지 갯수
    CHAT_TTL: int = 60 * 60 * 6  # TTL (초)
    CHAT_INIT_LOCK_TTL_SECONDS: int = 5  # 첫인사 생성 분산 락 TTL (초)
    AGENT_REQUEST_LIMIT: int = 6  # 요청당 모델 최대 호출 횟수
    TOOL_TIMEOUT_SECONDS: float = 15.0  # 도구 대기 시간 (초)
    TOOL_HISTORY_MAX_CHARS: int = 300  # 히스토리 저장 시 도구 결과 최대 길이
    LLM_TIMEOUT_MS: int = 25_000  # 모델 호출 타임아웃 (밀리초)
    LLM_RETRY_ATTEMPTS: int = 2  # 모델 호출 시도 횟수 (최초 1회 + 재시도 1회)

    # 파일 업로드 & 멀티모달
    MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024
    ALLOWED_GEMINI_FILE_URI_PREFIX: str = "https://generativelanguage.googleapis.com/"
    HISTORY_ATTACHMENT_TURNS: int = 2  # 과거 히스토리 중 FileData로 재전송할 최근 유저 턴 수

    # 개인화 추천 (Recommender - 옵션 B: Gemini Embedding + NumPy 서빙)
    REC_EMBEDDING_MODEL: str = "gemini-embedding-001"
    REC_TEXT_EMBED_DIM: int = 768
    REC_EMBED_DIM: int = 128
    REC_CANDIDATE_POOL_SIZE: int = 50
    REC_CANDIDATE_TTL: int = 60 * 10  # 후보군 캐시 TTL (10분)
    REC_CANDIDATE_LOCK_TTL_SECONDS: int = 5  # F5 Cache Stampede 분산 락 TTL
    REC_POPULAR_CACHE_TTL: int = 60 * 3  # 인기 목록 캐시 TTL (3분)
    REC_QUERY_CACHE_TTL: int = 60 * 30  # 시맨틱 검색 쿼리 임베딩 캐시 TTL (30분)
    REC_USER_TASTE_TTL: int = 60 * 60 * 24  # 유저 취향 Redis 캐시 TTL (24시간)
    REC_WRITEBACK_INTERVAL_SECONDS: int = 60 * 15  # Write-Back DB 백업 주기 (15분)
    REC_CATALOG_SYNC_INTERVAL_SECONDS: int = 60 * 60  # 카탈로그 자동 재동기화 주기 (1시간)
    REC_AUTO_SYNC_ON_STARTUP: bool = True  # 기동 시 아티팩트 없으면 자동 동기화
    REC_MAX_ARTIST_PER_PAGE: int = 2  # 화면 내 같은 아티스트 최대 노출 수 (Diversity Cap)
    REC_MAX_STICKY_ARTIST_PER_PAGE: int = 3  # 반복 진입(Sticky) 아티스트 허용 상한


constant = Constants()

# 로거
logger = logging.getLogger("uvicorn.error")
