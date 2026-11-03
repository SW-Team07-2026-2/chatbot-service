from redis.asyncio import Redis, from_url
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from app.config import logger, settings

__all__ = [
    "get_client",
    "ping",
    "close_client",
]

_redis: Redis | None = None

def get_client() -> Redis:
    """Redis 클라이언트 갯

    Returns:
        redis.asyncio.Redis
    """
    global _redis
    if _redis is None:

        _redis = from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20, # 커넥션 풀 최대 연결 n
            retry_on_timeout=True,
            retry=Retry(
                backoff=ExponentialBackoff(),
                retries=3, # 연결 재시도 n
            ),
        )
    return _redis


async def close_client() -> None:
    """Redis 연결 종료"""
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception as e:
            pass        
        finally:
            _redis = None


async def ping() -> bool:
    """Redis 정상인지?

    Returns:
        True: 정상
        False: 장애
    """
    try:
        return bool(await get_client().ping())
    except Exception:
        return False