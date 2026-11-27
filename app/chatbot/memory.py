import asyncio
from datetime import datetime
import json
import time
from typing import Any

import httpx

from app.config import constant, logger, setting, to_kst
from app.server.auth import get_current_user_auth, get_current_user_uuid
from app.server.http import get_backend_client
from app.server.redis import get_client

__all__ = ["add_user_memory", "get_user_memories"]


def _make_ltm_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:ltm"


def _make_loaded_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:ltm:loaded"


def _make_lock_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:ltm:lock"


def _build_backend_headers(
    include_csrf: bool = False,
    auth: tuple[str, str] | None = None,
) -> dict[str, str]:
    cookie_header, xsrf_token = auth if auth is not None else get_current_user_auth()
    headers: dict[str, str] = {}

    if setting.BACKEND_HOST.strip():
        headers["Host"] = setting.BACKEND_HOST.strip()
    if cookie_header:
        headers["Cookie"] = cookie_header
    if include_csrf and xsrf_token:
        headers["X-XSRF-TOKEN"] = xsrf_token

    return headers


def _parse_spoken_at(spoken_at: str | None, fallback: float) -> float:
    if not spoken_at:
        return fallback
    try:
        return datetime.fromisoformat(spoken_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return fallback


def _format_memory_entry(text: str, ts: float | None) -> str:
    """장기 기억 문장 앞에 한국 시각(KST) 포맷을 붙여 반환"""
    if ts is None:
        return text
    dt = to_kst(ts)
    return f"{dt.year}년 {dt.month}월 {dt.day}일 {dt.hour:02d}시 {dt.minute:02d}분: {text}"


def _extract_memories_from_response(data: Any) -> list[tuple[str, float]]:
    if not isinstance(data, dict) or data.get("isSuccess") is False:
        return []
    result = data.get("result")
    if not isinstance(result, dict):
        return []
    messages = result.get("messages")
    if not isinstance(messages, list):
        return []

    base_ts = time.time()
    items: list[tuple[str, float]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        text = str(msg.get("content", "")).strip()
        if not text:
            continue
        ts = _parse_spoken_at(msg.get("spokenAt"), base_ts) + idx * 1e-6
        items.append((text, ts))
    return items


async def _cache_ltm_queue(user_uuid: str, items: list[tuple[str, float]]) -> None:
    """백엔드에서 조회한 기억 목록을 Redis ZSET(50개 FIFO 큐) + 센티널 키에 적재"""
    key = _make_ltm_key(user_uuid)
    loaded_key = _make_loaded_key(user_uuid)
    trimmed = items[-constant.LTM_MAX :]

    try:
        r = get_client()
        async with r.pipeline(transaction=True) as p:
            p.delete(key)
            if trimmed:
                mapping = {
                    json.dumps({"t": ts, "m": text}, ensure_ascii=False): ts
                    for text, ts in trimmed
                }
                p.zadd(key, mapping)
                p.zremrangebyrank(key, 0, -(constant.LTM_MAX + 1))
                p.expire(key, constant.LTM_TTL)
            p.set(loaded_key, "1", ex=constant.LTM_TTL)
            await p.execute()
    except Exception as e:
        logger.warning(f"[장기 기억 캐싱 실패] ({user_uuid}): {e}")


async def _append_ltm_to_redis(user_uuid: str, content: str, ts: float) -> None:
    """Redis ZSET에 새 기억 1건 추가 및 상한 유지"""
    key = _make_ltm_key(user_uuid)
    loaded_key = _make_loaded_key(user_uuid)
    member = json.dumps({"t": ts, "m": content}, ensure_ascii=False)

    try:
        r = get_client()
        async with r.pipeline(transaction=True) as p:
            p.zadd(key, {member: ts})
            p.zremrangebyrank(key, 0, -(constant.LTM_MAX + 1))
            p.expire(key, constant.LTM_TTL)
            p.set(loaded_key, "1", ex=constant.LTM_TTL)
            await p.execute()
    except Exception as e:
        logger.warning(f"[장기 기억 Redis 반영 실패] ({user_uuid}): {e}")


async def _fetch_from_backend(
    user_uuid: str,
    auth: tuple[str, str] | None = None,
) -> list[tuple[str, float]] | None:
    headers = _build_backend_headers(include_csrf=False, auth=auth)
    try:
        res = await get_backend_client().get(
            constant.LTM_HISTORY_URL,
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        logger.error(f"[장기 기억 백엔드 조회 연결 실패] ({user_uuid}): {e}")
        return None

    if res.status_code != 200:
        logger.error(f"[장기 기억 백엔드 조회 실패] ({user_uuid}) status={res.status_code}")
        return None

    try:
        data = res.json()
    except Exception as e:
        logger.error(f"[장기 기억 백엔드 응답 파싱 실패] ({user_uuid}): {e}")
        return None

    if not isinstance(data, dict) or data.get("isSuccess") is False:
        logger.error(f"[장기 기억 백엔드 조회 거부] ({user_uuid}): {data}")
        return None

    return _extract_memories_from_response(data)


async def _read_ltm_from_redis(user_uuid: str) -> list[str] | None:
    """Redis 센티널 및 ZSET 조회 (캐시 미스 시 None 반환)"""
    key = _make_ltm_key(user_uuid)
    loaded_key = _make_loaded_key(user_uuid)

    try:
        r = get_client()
        if not await r.exists(loaded_key):
            return None

        raw_items = await r.zrange(key, 0, -1)
        async with r.pipeline(transaction=True) as p:
            p.expire(loaded_key, constant.LTM_TTL)
            if raw_items:
                p.expire(key, constant.LTM_TTL)
            await p.execute()

        memories: list[str] = []
        for raw in raw_items:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("m"):
                    raw_ts = parsed.get("t")
                    ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else None
                    memories.append(_format_memory_entry(str(parsed["m"]), ts))
            except Exception:
                continue
        return memories
    except Exception as e:
        logger.warning(f"[장기 기억 Redis 조회 실패] ({user_uuid}): {e}")
        return None


async def add_user_memory(
    content: str,
    user_uuid: str | None = None,
    auth: tuple[str, str] | None = None,
) -> bool:
    """새 장기 기억을 백엔드 DB에 먼저 저장하고, 성공 시에만 Redis ZSET에 반영한다."""
    uid = (user_uuid if user_uuid is not None else get_current_user_uuid()).strip()
    text = str(content or "").strip()
    if not uid or not text:
        return False

    if len(text) > constant.LTM_MAX_CONTENT_LENGTH:
        text = text[: constant.LTM_MAX_CONTENT_LENGTH].rstrip()

    now_dt = to_kst()
    now_ts = now_dt.timestamp()
    payload = {
        "messages": [
            {
                "role": "USER",
                "content": text,
                "spokenAt": now_dt.isoformat(timespec="seconds"),
            }
        ]
    }
    headers = _build_backend_headers(include_csrf=True, auth=auth)

    try:
        res = await get_backend_client().post(
            constant.LTM_HISTORY_URL,
            json=payload,
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as e:
        logger.error(f"[장기 기억 백엔드 저장 연결 실패] ({uid}): {e}")
        return False

    if res.status_code != 200:
        logger.error(f"[장기 기억 백엔드 저장 실패] ({uid}) status={res.status_code}")
        return False

    try:
        data = res.json()
    except Exception as e:
        logger.error(f"[장기 기억 백엔드 저장 응답 파싱 실패] ({uid}): {e}")
        return False

    if not isinstance(data, dict) or data.get("isSuccess") is False:
        logger.error(f"[장기 기억 백엔드 저장 거부] ({uid}): {data}")
        return False

    await _append_ltm_to_redis(uid, text, now_ts)
    logger.info(f"[장기 기억 저장 완료] ({uid}): {text}")
    return True


async def get_user_memories(
    user_uuid: str | None = None,
    auth: tuple[str, str] | None = None,
) -> list[str]:
    """Redis 캐시를 우선 조회하고, 캐시 미스 시 백엔드에서 가져와 50개 큐(ZSET)로 적재한다."""
    uid = (user_uuid if user_uuid is not None else get_current_user_uuid()).strip()
    if not uid:
        return []

    cached = await _read_ltm_from_redis(uid)
    if cached is not None:
        return cached

    lock_key = _make_lock_key(uid)
    lock_acquired = False
    try:
        lock_acquired = bool(
            await get_client().set(
                lock_key, "1", nx=True, ex=constant.LTM_LOCK_TTL_SECONDS
            )
        )
    except Exception:
        lock_acquired = False

    if not lock_acquired:
        for _ in range(3):
            await asyncio.sleep(0.15)
            cached = await _read_ltm_from_redis(uid)
            if cached is not None:
                return cached

    try:
        items = await _fetch_from_backend(uid, auth=auth)
        if items is None:
            return []

        await _cache_ltm_queue(uid, items)
        return [_format_memory_entry(text, ts) for text, ts in items[-constant.LTM_MAX :]]
    finally:
        if lock_acquired:
            try:
                await get_client().delete(lock_key)
            except Exception:
                pass
