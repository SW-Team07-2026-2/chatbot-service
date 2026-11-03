from dataclasses import dataclass, field
import json
from typing import List, Optional

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

from app.config import logger, settings
from app.ai.agent import SESSION_MAX_MESSAGES, SESSION_TTL_SECONDS


class RedisServiceUnavailable(RuntimeError):
    """Redis 세션 저장소가 오프라인이거나 연결할 수 없을 때 발생하는 예외"""
    pass


@dataclass
class ConversationSession:
    """유저 대화 세션 및 메타데이터"""
    messages: List[ModelMessage] = field(default_factory=list)
    total_count: int = 0
    has_truncated: bool = False


# Redis 연결 인스턴스 (싱글톤)
_redis: Optional[Redis] = None


def get_redis_client() -> Optional[Redis]:
    """
    Redis 클라이언트 가져오기
    """
    global _redis
    url = settings.REDIS_URL
    if _redis is None:
        _redis = from_url(url, decode_responses=False)
    return _redis


async def check_redis_connection() -> None:
    """Redis 연결 상태를 확인하고, 꺼져있으면 예외를 발생시켜 서버 시작을 즉시 중단한다."""
    url = settings.REDIS_URL
    client = get_redis_client()
    if client is None:
        raise RuntimeError("[REDIS] Redis 클라이언트를 생성할 수 없습니다.") from None

    try:
        await client.ping()
        logger.info(f"Redis 연결 성공 ({url})")
    except Exception:
        raise RuntimeError(
            f"[REDIS] Redis({url})에 연결할 수 없습니다. Docker 또는 Redis 서버를 켜주세요."
        ) from None


async def close_redis_connection() -> None:
    """서버 종료 시 Redis 커넥션 닫기"""
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None


def _make_key(user_id: str) -> str:
    return f"session:{user_id}"


async def get_conversation_session(user_id: str) -> ConversationSession:
    """Redis에서 유저의 세션(메시지 목록 + 누적 카운트 + 잘림 여부) 조회"""
    client = get_redis_client()
    if client is None:
        raise RedisServiceUnavailable("Redis 클라이언트가 초기화되지 않았습니다.")

    key = _make_key(user_id)
    try:
        raw_data = await client.get(key)
    except (RedisError, ConnectionRefusedError, OSError) as e:
        logger.error(f"[REDIS 연결 불가] 대화 기록 조회 실패 ({user_id}): {e}")
        raise RedisServiceUnavailable(
            f"세션 저장소(Redis)에 연결할 수 없습니다: {e}"
        ) from e

    if not raw_data:
        return ConversationSession()

    # 신규 포맷(dict: {total_count, has_truncated, messages}) 또는 레거시 포맷(list) 호환
    try:
        parsed = json.loads(raw_data)
        if isinstance(parsed, dict) and "messages" in parsed:
            messages = ModelMessagesTypeAdapter.validate_python(parsed["messages"])
            total_count = parsed.get("total_count", len(messages))
            has_truncated = parsed.get(
                "has_truncated", total_count > SESSION_MAX_MESSAGES
            )
            return ConversationSession(
                messages=messages,
                total_count=total_count,
                has_truncated=has_truncated,
            )
    except Exception as parse_err:
        logger.warning(f"신규 포맷 파싱 실패, 레거시 포맷으로 재시도: {parse_err}")

    # 레거시 포맷 fallback
    try:
        messages = ModelMessagesTypeAdapter.validate_json(raw_data)
        return ConversationSession(
            messages=messages,
            total_count=len(messages),
            has_truncated=len(messages) >= SESSION_MAX_MESSAGES,
        )
    except Exception as parse_err:
        logger.warning(f"대화 기록 역직렬화 실패, 빈 세션으로 초기화: {parse_err}")
        return ConversationSession()


async def get_conversation_history(user_id: str) -> List[ModelMessage]:
    """Redis에서 유저의 최근 대화 기록 조회 (기존 호환용)"""
    session = await get_conversation_session(user_id)
    return session.messages


async def save_conversation_history(
    user_id: str,
    messages: List[ModelMessage],
    prev_total_count: int = 0,
    prev_history_len: int = 0,
) -> ConversationSession:
    """최근 대화 기록(최대 10턴/20메시지)과 누적 카운트를 Redis에 1시간 만료(TTL)로 갱신 저장"""
    client = get_redis_client()
    if client is None:
        raise RedisServiceUnavailable("Redis 클라이언트가 초기화되지 않았습니다.")

    key = _make_key(user_id)
    # 이번 턴에 실제로 추가된 메시지 수 계산
    added_count = max(0, len(messages) - prev_history_len)
    total_count = prev_total_count + added_count
    if total_count == 0:
        total_count = len(messages)

    has_truncated = total_count > SESSION_MAX_MESSAGES
    trimmed = messages[-SESSION_MAX_MESSAGES:]

    payload_dict = {
        "total_count": total_count,
        "has_truncated": has_truncated,
        "messages": ModelMessagesTypeAdapter.dump_python(trimmed, mode="json"),
    }
    payload = json.dumps(payload_dict)

    try:
        await client.set(key, payload, ex=SESSION_TTL_SECONDS)
        return ConversationSession(
            messages=trimmed,
            total_count=total_count,
            has_truncated=has_truncated,
        )
    except (RedisError, ConnectionRefusedError, OSError) as e:
        logger.error(f"[REDIS 연결 불가] 대화 기록 저장 실패 ({user_id}): {e}")
        raise RedisServiceUnavailable(
            f"세션 저장소(Redis)에 연결할 수 없습니다: {e}"
        ) from e


async def clear_conversation_history(user_id: str) -> bool:
    """Redis에서 유저의 대화 기록 삭제 (세션 초기화)"""
    client = get_redis_client()
    if client is None:
        raise RedisServiceUnavailable("Redis 클라이언트가 초기화되지 않았습니다.")

    key = _make_key(user_id)
    try:
        await client.delete(key)
        return True
    except (RedisError, ConnectionRefusedError, OSError) as e:
        logger.error(f"[REDIS 연결 불가] 대화 기록 삭제 실패 ({user_id}): {e}")
        raise RedisServiceUnavailable(
            f"세션 저장소(Redis)에 연결할 수 없습니다: {e}"
        ) from e
