import asyncio
import json
import random
import time
from typing import Any
import uuid

from google.genai import types

from app.config import constant, logger, to_kst
from app.server.redis import get_client
from .agent import GREETING


class RedisServiceUnavailable(RuntimeError):
    """Redis 세션 저장소 연결 불가 예외"""
    pass


# Redis 대화 기록 관리 =====================================================================

def make_chat_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:chat"


def make_chat_init_lock_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:chat:init_lock"


async def get_raw_chat_history(user_uuid: str) -> list[dict]:
    """Redis List에서 전체 기록 조회 (기록 없으면 분산 락으로 첫 인사 1회만 생성 후 저장)"""
    try:
        r = get_client()
        key = make_chat_key(user_uuid)
        raw_items = await r.lrange(key, 0, -1)

        if not raw_items:
            lock_key = make_chat_init_lock_key(user_uuid)
            lock_acquired = bool(
                await r.set(
                    lock_key,
                    "1",
                    nx=True,
                    ex=constant.CHAT_INIT_LOCK_TTL_SECONDS,
                )
            )
            if lock_acquired:
                try:
                    raw_items = await r.lrange(key, 0, -1)
                    if not raw_items:
                        first_greeting = random.choice(GREETING)
                        await append_chat_message(user_uuid, "bot", first_greeting)
                        raw_items = await r.lrange(key, 0, -1)
                finally:
                    try:
                        await r.delete(lock_key)
                    except Exception:
                        pass
            else:
                for _ in range(3):
                    await asyncio.sleep(0.1)
                    raw_items = await r.lrange(key, 0, -1)
                    if raw_items:
                        break
                if not raw_items:
                    first_greeting = random.choice(GREETING)
                    await append_chat_message(user_uuid, "bot", first_greeting)
                    raw_items = await r.lrange(key, 0, -1)

        return [json.loads(x) for x in raw_items if x]
    except RedisServiceUnavailable:
        raise
    except Exception as e:
        logger.error(f"[Redis 에러] 기록 조회 실패 ({user_uuid}): {e}")
        raise RedisServiceUnavailable(f"Redis 연결 실패: {e}") from e


async def append_chat_message(user_uuid: str, role: str, content: str) -> None:
    """단일 메시지 RPUSH + LTRIM + EXPIRE 원자적 저장"""
    await append_chat_items(user_uuid, [{"role": role, "content": content, "ts": time.time()}])


async def append_chat_items(user_uuid: str, items: list[dict]) -> None:
    """여러 대화/도구 항목을 시간순으로 한 번에 RPUSH + LTRIM + EXPIRE 저장"""
    if not items:
        return
    key = make_chat_key(user_uuid)
    normalized: list[dict] = []
    for item in items:
        record = dict(item)
        if record.get("role") in ("user", "bot") and not record.get("id"):
            record["id"] = f"msg-{uuid.uuid4().hex[:12]}"
        normalized.append(record)
    serialized = [json.dumps(item, ensure_ascii=False) for item in normalized]
    try:
        r = get_client()
        async with r.pipeline(transaction=True) as p:
            p.rpush(key, *serialized)
            p.ltrim(key, -constant.CHAT_MAX, -1)
            p.expire(key, constant.CHAT_TTL)
            await p.execute()
    except Exception as e:
        logger.error(f"[Redis 에러] 메시지 저장 실패 ({user_uuid}): {e}")
        raise RedisServiceUnavailable(f"Redis 저장 실패: {e}") from e


async def save_turn_history(
    user_uuid: str,
    message_text: str,
    valid_attachments: list[dict],
    recorded_tools: list[dict],
    bot_text: str,
) -> None:
    """현재 턴의 (user -> tool -> bot) 기록을 Redis에 일괄 저장"""
    now_ts = time.time()
    user_record: dict[str, Any] = {"role": "user", "content": message_text, "ts": now_ts}
    if valid_attachments:
        user_record["attachments"] = valid_attachments

    items_to_save = [
        user_record,
        *recorded_tools,
        {"role": "bot", "content": bot_text, "ts": now_ts},
    ]
    await append_chat_items(user_uuid, items_to_save)


# 포맷 변환 ==================================================================================

def create_uploaded_file_part(att: dict) -> types.Part | None:
    """첨부 메타데이터에서 Gemini 파일 Part 생성 (허용된 Gemini File URI만 통과)"""
    uri = str(att.get("file_uri") or att.get("url") or "").strip()
    if not uri or not uri.startswith(constant.ALLOWED_GEMINI_FILE_URI_PREFIX):
        return None
    media_type = att.get("media_type") or att.get("type")
    return types.Part(file_data=types.FileData(file_uri=uri, mime_type=media_type))


def compact_tool_result_for_history(result: Any) -> Any:
    """Redis 히스토리 저장용 도구 결과 경량화 (일회성 지시문 제거 및 과도한 본문 축약)"""
    max_chars = constant.TOOL_HISTORY_MAX_CHARS

    def _truncate(val: Any) -> Any:
        if isinstance(val, str) and len(val) > max_chars:
            return val[:max_chars].rstrip() + "..."
        return val

    if isinstance(result, dict):
        return {
            k: _truncate(v)
            for k, v in result.items()
            if k != "message_for_ai"
        }
    return _truncate(result)


def to_function_response(result: Any) -> dict:
    """도구 결과 -> Gemini function_response dict 변환"""
    return result if isinstance(result, dict) else {"result": result}


def korea_time(ts: float | None) -> str:
    """timestamp -> 한국 표준시 '오전/오후 H:MM' 변환 (프론트 UI용)"""
    dt = to_kst(ts)
    period = "오후" if dt.hour >= 12 else "오전"
    hour = dt.hour % 12 or 12
    return f"{period} {hour}:{dt.minute:02d}"


def format_model_time(ts: float | None = None) -> str:
    """timestamp -> 'M월 D일 HH:MM' 변환 (LLM 시간 인지용)"""
    dt = to_kst(ts)
    return f"{dt.month}월 {dt.day}일 {dt.hour:02d}:{dt.minute:02d}"


def raw_history_to_frontend(raw_history: list[dict]) -> list[dict]:
    """Redis 기록 -> 프론트용 json 변환"""
    formatted: list[dict] = []
    for idx, item in enumerate(raw_history, start=1):
        role = item.get("role", "user")
        if role not in ("user", "bot"):
            continue

        content = str(item.get("content", "")).strip()
        msg_data: dict[str, Any] = {
            "id": str(item.get("id") or f"hist-{role}-{idx}"),
            "role": role,
            "text": content,
            "time": korea_time(item.get("ts")),
        }
        if item.get("attachments"):
            msg_data["attachments"] = item["attachments"]

        formatted.append(msg_data)
    return formatted


def raw_history_to_contents(raw_history: list[dict]) -> list[types.Content]:
    """Redis 기록 -> Gemini Content 리스트 복원 (tool 기록 및 최근 N턴 멀티모달 파일 복원)"""
    user_indices = [
        idx
        for idx, item in enumerate(raw_history)
        if item.get("role", "user") == "user"
        and (str(item.get("content", "")).strip() or item.get("attachments"))
    ]
    recent_file_turn_indices = (
        set(user_indices[-constant.HISTORY_ATTACHMENT_TURNS :])
        if constant.HISTORY_ATTACHMENT_TURNS > 0
        else set()
    )

    contents: list[types.Content] = []
    for idx, item in enumerate(raw_history):
        role = item.get("role", "user")
        ts = item.get("ts", time.time())

        if role == "user":
            content = str(item.get("content", "")).strip()
            raw_attachments = item.get("attachments") or []
            if not content and not raw_attachments:
                continue

            timed_content = (
                f"[{format_model_time(ts)}] {content}"
                if content
                else f"[{format_model_time(ts)}] (첨부 파일 확인 요청)"
            )

            parts: list[types.Part] = []
            if idx in recent_file_turn_indices:
                parts.append(types.Part(text=timed_content))
                for att in raw_attachments:
                    if isinstance(att, dict):
                        fp = create_uploaded_file_part(att)
                        if fp:
                            parts.append(fp)
            else:
                file_names = [
                    str(att.get("file_name") or att.get("name") or "첨부파일").strip()
                    for att in raw_attachments
                    if isinstance(att, dict)
                ]
                if file_names:
                    timed_content = f"{timed_content}\n[이전 첨부파일: {', '.join(file_names)}]"
                parts.append(types.Part(text=timed_content))

            contents.append(types.Content(role="user", parts=parts))

        elif role == "tool":
            tool_name = item.get("tool", "unknown_tool")
            tool_args = item.get("args") or {}
            tool_content = item.get("content", "")

            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(name=tool_name, args=tool_args)],
                )
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name, response=to_function_response(tool_content)
                        )
                    ],
                )
            )

        elif role == "bot":
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

    def is_plain_user(c: types.Content) -> bool:
        return c.role == "user" and not any(p.function_response for p in (c.parts or []))

    # 첫 메시지가 순수 유저 질문이 될 때까지 앞쪽 잔여 메시지 제거 (API 거절 방지)
    while contents and not is_plain_user(contents[0]):
        contents.pop(0)

    # 끝 메시지가 AI 응답(model)으로 끝나도록 정리
    while contents and contents[-1].role != "model":
        contents.pop(-1)

    return contents

