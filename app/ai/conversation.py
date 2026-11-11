import random
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

from app.config import logger
from app.server.redis import get_client
from .agent import (
    AGENT_REQUEST_LIMIT,
    CHAT_MAX,
    CHAT_TTL,
    GREETING,
    TOOL_STATUS_MESSAGES,
    agent,
)

USAGE_LIMITS = UsageLimits(request_limit=AGENT_REQUEST_LIMIT)
KST = timezone(timedelta(hours=9))

class RedisServiceUnavailable(RuntimeError):
    """Redis 세션 저장소 연결 불가 예외"""
    pass

#===============================================================================================================================================
# Redis 대화 기록 관리

def make_chat_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:chat"


async def get_raw_chat_history(user_uuid: str) -> list[dict]:
    """Redis List에서 전체 기록 조회 (기록 없으면 첫 인사 생성 후 저장)"""
    try:
        r = get_client()
        key = make_chat_key(user_uuid)
        raw_items = await r.lrange(key, 0, -1)

        # 기록이 없으면 첫인사를 생성하여 Redis에 저장
        if not raw_items:
            first_greeting = random.choice(GREETING)
            await append_chat_message(user_uuid, "bot", first_greeting)
            raw_items = await r.lrange(key, 0, -1)

        return [json.loads(x) for x in raw_items if x]
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
    serialized = [json.dumps(item, ensure_ascii=False) for item in items]
    try:
        r = get_client()
        async with r.pipeline(transaction=True) as p:
            p.rpush(key, *serialized)
            p.ltrim(key, -CHAT_MAX, -1)
            p.expire(key, CHAT_TTL)
            await p.execute()
    except Exception as e:
        logger.error(f"[Redis 에러] 메시지 저장 실패 ({user_uuid}): {e}")
        raise RedisServiceUnavailable(f"Redis 저장 실패: {e}") from e


#===============================================================================================================================================
# 포맷 변환

def create_uploaded_file_part(att: dict) -> UploadedFile | None:
    """첨부 메타데이터에서 PydanticAI UploadedFile 파트 생성"""
    uri = att.get("file_uri") or att.get("url")
    if not uri:
        return None
    media_type = att.get("media_type") or att.get("type")
    return UploadedFile(
        file_id=uri,
        provider_name="google",
        media_type=media_type,
    )


def korea_time(ts: float | None) -> str:
    """timestamp -> 한국 표준시 '오전/오후 H:MM' 변환 (프론트 UI용)"""
    dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).astimezone(KST)
    period = "오후" if dt.hour >= 12 else "오전"
    hour = dt.hour % 12 or 12
    return f"{period} {hour}:{dt.minute:02d}"


def format_model_time(ts: float | None = None) -> str:
    """timestamp -> 'M월 D일 HH:MM' 변환 (LLM 시간 인지용)"""
    dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).astimezone(KST)
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
            "id": f"hist-{role}-{idx}",
            "role": role,
            "text": content,
            "time": korea_time(item.get("ts")),
        }
        if "attachments" in item and item["attachments"]:
            msg_data["attachments"] = item["attachments"]

        formatted.append(msg_data)
    return formatted


def raw_history_to_model_messages(raw_history: list[dict]) -> list[ModelMessage]:
    """Redis 기록 -> pydantic_ai ModelMessage 리스트 복원 (tool 기록 및 멀티모달 파일 복원)"""
    model_msgs: list[ModelMessage] = []
    for idx, item in enumerate(raw_history):
        role = item.get("role", "user")
        ts = item.get("ts", time.time())
        ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        # 유저 질문 (첨부 파일이 있을 경우 UploadedFile 함께 복원)
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

            file_parts: list[UploadedFile] = []
            for att in raw_attachments:
                if isinstance(att, dict):
                    uf = create_uploaded_file_part(att)
                    if uf:
                        file_parts.append(uf)

            prompt_content: Any = (
                [timed_content, *file_parts] if file_parts else timed_content
            )

            model_msgs.append(
                ModelRequest(
                    parts=[UserPromptPart(content=prompt_content, timestamp=ts_dt)],
                    timestamp=ts_dt,
                )
            )

        # 도구 호출 및 실행 결과 (Gemini 전송 시 모델 호출 -> 유저(functionResponse) 턴으로 자동 매핑됨)
        elif role == "tool":
            tool_name = item.get("tool", "unknown_tool")
            tool_args = item.get("args", {})
            tool_content = item.get("content", "")
            call_id = f"hist_call_{idx}"

            model_msgs.append(
                ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=call_id)],
                    timestamp=ts_dt,
                )
            )
            model_msgs.append(
                ModelRequest(
                    parts=[ToolReturnPart(tool_name=tool_name, content=tool_content, tool_call_id=call_id, timestamp=ts_dt)],
                    timestamp=ts_dt,
                )
            )

        # 봇 최종 텍스트 답변
        elif role == "bot":
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            model_msgs.append(ModelResponse(parts=[TextPart(content=content)], timestamp=ts_dt))

    # 첫 메시지가 순수 유저 질문(UserPromptPart)이 될 때까지 앞쪽 잔여 메시지 제거 (API 거절 방지)
    while model_msgs and not (
        isinstance(model_msgs[0], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in getattr(model_msgs[0], "parts", []))
    ):
        model_msgs.pop(0)

    # 끝 메시지가 AI 응답(ModelResponse)으로 끝나도록 정리
    while model_msgs and not isinstance(model_msgs[-1], ModelResponse):
        model_msgs.pop(-1)

    return model_msgs

#===============================================================================================================================================
# 실시간 대화 스트리밍

async def stream_conversation_events(
    user_uuid: str,
    message: str,
    attachments: list[dict] | None = None,
    is_disconnected_check: Callable[[], Coroutine[Any, Any, bool]] | None = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """실시간 대화 이벤트 스트리밍 제너레이터 (텍스트 및 멀티모달 파일 지원)"""
    message_text = str(message or "").strip()
    valid_attachments = [att for att in (attachments or []) if isinstance(att, dict) and (att.get("file_uri") or att.get("url"))]

    # 빈 메시지 및 무효 첨부 방어
    if not message_text and not valid_attachments:
        return

    # 대화 기록 조회 및 모델 컨텍스트 복원 (user / tool / bot 모두 포함)
    raw_history = await get_raw_chat_history(user_uuid)
    model_msgs = raw_history_to_model_messages(raw_history)

    yield (
        "status",
        {
            "status": "thinking",
            "message": "답변을 생각하고 있어요...",
        },
    )

    accumulated_bot_text: list[str] = []
    pending_action: dict | None = None
    tool_calls_map: dict[str, dict] = {}
    recorded_tools: list[dict] = []

    # 멀티모달 파일 파트 생성
    file_parts: list[UploadedFile] = []
    for att in valid_attachments:
        uf = create_uploaded_file_part(att)
        if uf:
            file_parts.append(uf)

    # 유저 질문 프롬프트 구성 (시간 정보 포함)
    timed_text = (
        f"[{format_model_time()}] {message_text}"
        if message_text
        else f"[{format_model_time()}] 첨부된 파일을 확인하고 분석해 줘."
    )
    prompt_input: Any = [timed_text, *file_parts] if file_parts else timed_text

    try:
        async with agent.run_stream_events(
            prompt_input,
            message_history=model_msgs,
            deps=user_uuid,
            usage_limits=USAGE_LIMITS,
        ) as events:
            async for event in events:
                if is_disconnected_check and await is_disconnected_check():
                    logger.warning(f"클라이언트 연결 끊김 ({user_uuid})")
                    break

                # 도구 호출 시작
                if isinstance(event, FunctionToolCallEvent):
                    tool_name = getattr(event.part, "tool_name", "")
                    call_id = getattr(event.part, "tool_call_id", "")
                    tool_args = (
                        event.part.args_as_dict()
                        if hasattr(event.part, "args_as_dict")
                        else getattr(event.part, "args", {})
                    )
                    tool_calls_map[call_id] = {"tool": tool_name, "args": tool_args}

                    status_msg = TOOL_STATUS_MESSAGES.get(tool_name, f"'{tool_name}' 작업 진행 중...")
                    yield (
                        "status",
                        {
                            "status": "tool_calling",
                            "tool": tool_name,
                            "message": status_msg,
                        },
                    )

                # 도구 실행 완료
                elif isinstance(event, FunctionToolResultEvent):
                    tool_name = getattr(event.part, "tool_name", "")
                    call_id = getattr(event.part, "tool_call_id", "")
                    tool_result = (
                        getattr(event, "result", None)
                        or getattr(event.part, "result", None)
                        or getattr(event.part, "content", None)
                    )
                    # 문자열로 직렬화되어 넘어오는 경우 JSON 파싱 시도
                    if isinstance(tool_result, str):
                        try:
                            tool_result = json.loads(tool_result)
                        except Exception:
                            pass

                    # action 키가 있으면 프론트엔드 전송용으로 보관 (message_for_ai 제외)
                    if isinstance(tool_result, dict) and "action" in tool_result:
                        pending_action = {k: v for k, v in tool_result.items() if k != "message_for_ai"}

                    # Redis 단일 히스토리 저장용 도구 실행 기록 보관
                    call_info = tool_calls_map.get(call_id, {"tool": tool_name, "args": {}})
                    recorded_tools.append({
                        "role": "tool",
                        "tool": tool_name,
                        "args": call_info["args"],
                        "content": tool_result,
                        "ts": time.time(),
                    })

                    yield (
                        "status",
                        {
                            "status": "thinking",
                            "tool": tool_name,
                            "message": "결과를 바탕으로 답변을 정리하고 있어요...",
                        },
                    )

                # 첫 텍스트 청크
                elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    content = str(event.part.content or "")
                    if content:
                        accumulated_bot_text.append(content)
                        yield ("text_delta", {"content": content})

                # 후속 텍스트 델타 청크
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    content = str(event.delta.content_delta or "")
                    if content:
                        accumulated_bot_text.append(content)
                        yield ("text_delta", {"content": content})

        # 말이 다 끝난 직후 프론트엔드 액션 방출 (말풍선 하단 버튼 렌더링용)
        if pending_action:
            yield ("action", pending_action)

        # 스트리밍 완료 후 Redis 단일 저장소에 (user -> tool -> bot) 시간순 일괄 저장
        full_bot_response = "".join(accumulated_bot_text).strip()
        if full_bot_response:
            now_ts = time.time()
            user_record: dict[str, Any] = {"role": "user", "content": message_text, "ts": now_ts}
            if valid_attachments:
                user_record["attachments"] = valid_attachments

            items_to_save = [
                user_record,
                *recorded_tools,
                {"role": "bot", "content": full_bot_response, "ts": now_ts},
            ]
            await append_chat_items(user_uuid, items_to_save)

        yield ("done", {"status": "completed"})

    except UsageLimitExceeded:
        fallback_msg = "요청을 처리하지 못했어요. 조금 더 간단하게 다시 말씀해 주시겠어요?"
        partial = "".join(accumulated_bot_text).strip()
        saved_bot_msg = f"{partial}\n\n{fallback_msg}" if partial else fallback_msg
        try:
            now_ts = time.time()
            user_record = {"role": "user", "content": message_text, "ts": now_ts}
            if valid_attachments:
                user_record["attachments"] = valid_attachments
            await append_chat_items(
                user_uuid,
                [user_record, *recorded_tools, {"role": "bot", "content": saved_bot_msg, "ts": now_ts}],
            )
        except Exception as save_err:
            logger.error(f"[Redis 에러] 제한 초과 턴 저장 실패 ({user_uuid}): {save_err}")
        yield ("text_delta", {"content": f"\n\n{fallback_msg}" if partial else fallback_msg})
        yield ("done", {"status": "limit_exceeded"})

    except RedisServiceUnavailable as e:
        logger.error(f"[Redis 장애] 스트리밍 저장 실패: {e}")
        yield (
            "error",
            {"error": "대화 기록 저장에 실패했습니다.", "code": "REDIS_UNAVAILABLE"},
        )

    except Exception as e:
        err_msg = str(e).lower()
        err_type = type(e).__name__.lower()
        partial = "".join(accumulated_bot_text).strip()
        if "timeout" in err_msg or "timeout" in err_type:
            logger.error(f"[AI 시간 초과] LLM 응답 대기 시간 초과 ({user_uuid}): {e}")
            timeout_msg = "답변 생성에 시간이 너무 오래 걸려 중단되었어요. 잠시 후 다시 말씀해 주시겠어요?"
            saved_bot_msg = f"{partial}\n\n{timeout_msg}" if partial else timeout_msg
            try:
                now_ts = time.time()
                user_record = {"role": "user", "content": message_text, "ts": now_ts}
                if valid_attachments:
                    user_record["attachments"] = valid_attachments
                await append_chat_items(
                    user_uuid,
                    [user_record, *recorded_tools, {"role": "bot", "content": saved_bot_msg, "ts": now_ts}],
                )
            except Exception as save_err:
                logger.error(f"[Redis 에러] 타임아웃 턴 저장 실패 ({user_uuid}): {save_err}")
            yield ("text_delta", {"content": f"\n\n{timeout_msg}" if partial else timeout_msg})
            yield ("done", {"status": "timeout"})
        else:
            logger.error(f"스트리밍 처리 에러 ({user_uuid}): {e}")
            err_fallback = "답변 생성 중 문제가 발생했습니다. 잠시 후 다시 말씀해 주시겠어요?"
            saved_bot_msg = f"{partial}\n\n{err_fallback}" if partial else err_fallback
            try:
                now_ts = time.time()
                user_record = {"role": "user", "content": message_text, "ts": now_ts}
                if valid_attachments:
                    user_record["attachments"] = valid_attachments
                await append_chat_items(
                    user_uuid,
                    [user_record, *recorded_tools, {"role": "bot", "content": saved_bot_msg, "ts": now_ts}],
                )
            except Exception as save_err:
                logger.error(f"[Redis 에러] 에러 턴 저장 실패 ({user_uuid}): {save_err}")
            yield ("error", {"error": "답변 생성 중 문제가 발생했습니다."})
