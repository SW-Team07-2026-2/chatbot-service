from collections.abc import AsyncGenerator, Callable, Coroutine
import time
from typing import Any

from google.genai import types

from app.config import constant, logger
from app.server.auth import set_current_user_uuid
from .agent import (
    MODEL_NAME,
    TOOL_STATUS_MESSAGES,
    UsageLimitExceeded,
    build_generate_config,
    build_system_instruction,
    execute_tool,
)
from .history import (
    RedisServiceUnavailable,
    compact_tool_result_for_history,
    create_uploaded_file_part,
    format_model_time,
    get_raw_chat_history,
    raw_history_to_contents,
    save_turn_history,
    to_function_response,
)
from .uploader import get_genai_client


async def stream_conversation_events(
    user_uuid: str,
    message: str,
    attachments: list[dict] | None = None,
    is_disconnected_check: Callable[[], Coroutine[Any, Any, bool]] | None = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """실시간 대화 이벤트 스트리밍 제너레이터 (텍스트 및 멀티모달 파일 지원)"""
    message_text = str(message or "").strip()
    valid_attachments = [
        att
        for att in (attachments or [])
        if isinstance(att, dict)
        and str(att.get("file_uri") or att.get("url") or "")
        .strip()
        .startswith(constant.ALLOWED_GEMINI_FILE_URI_PREFIX)
    ]

    if not message_text and not valid_attachments:
        return

    raw_history = await get_raw_chat_history(user_uuid)
    history_contents = raw_history_to_contents(raw_history)
    set_current_user_uuid(user_uuid)

    yield (
        "status",
        {
            "status": "thinking",
            "message": "답변을 생각하고 있어요...",
        },
    )

    accumulated_bot_text: list[str] = []
    pending_actions: list[dict] = []
    recorded_tools: list[dict] = []

    file_parts: list[types.Part] = []
    for att in valid_attachments:
        fp = create_uploaded_file_part(att)
        if fp:
            file_parts.append(fp)

    timed_text = (
        f"[{format_model_time()}] {message_text}"
        if message_text
        else f"[{format_model_time()}] 첨부된 파일을 확인하고 분석해 줘."
    )
    contents: list[types.Content] = [
        *history_contents,
        types.Content(role="user", parts=[types.Part(text=timed_text), *file_parts]),
    ]

    try:
        client = get_genai_client()
        config = build_generate_config(await build_system_instruction(user_uuid))
        disconnected = False
        finished = False

        for _ in range(constant.AGENT_REQUEST_LIMIT):
            function_calls: list[types.FunctionCall] = []
            model_parts: list[types.Part] = []

            stream = await client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                if is_disconnected_check and await is_disconnected_check():
                    logger.warning(f"클라이언트 연결 끊김 ({user_uuid})")
                    disconnected = True
                    break

                if not chunk.candidates:
                    continue
                candidate_content = chunk.candidates[0].content
                if not candidate_content or not candidate_content.parts:
                    continue

                for part in candidate_content.parts:
                    model_parts.append(part)
                    if part.function_call:
                        function_calls.append(part.function_call)
                    elif part.text and not part.thought:
                        accumulated_bot_text.append(part.text)
                        yield ("text_delta", {"content": part.text})

            if disconnected:
                break

            if not function_calls:
                finished = True
                break

            contents.append(types.Content(role="model", parts=model_parts))
            response_parts: list[types.Part] = []

            for fc in function_calls:
                tool_name = fc.name or ""
                tool_args = dict(fc.args or {})

                status_msg = TOOL_STATUS_MESSAGES.get(tool_name, f"'{tool_name}' 작업 진행 중...")
                yield (
                    "status",
                    {
                        "status": "tool_calling",
                        "tool": tool_name,
                        "message": status_msg,
                    },
                )

                tool_result = await execute_tool(tool_name, tool_args)

                if isinstance(tool_result, dict) and "action" in tool_result:
                    pending_actions.append(
                        {k: v for k, v in tool_result.items() if k != "message_for_ai"}
                    )

                recorded_tools.append({
                    "role": "tool",
                    "tool": tool_name,
                    "args": tool_args,
                    "content": compact_tool_result_for_history(tool_result),
                    "ts": time.time(),
                })

                response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name, response=to_function_response(tool_result)
                    )
                )

                yield (
                    "status",
                    {
                        "status": "thinking",
                        "tool": tool_name,
                        "message": "결과를 바탕으로 답변을 정리하고 있어요...",
                    },
                )

            contents.append(types.Content(role="user", parts=response_parts))

        if not finished and not disconnected:
            raise UsageLimitExceeded(f"모델 호출 {constant.AGENT_REQUEST_LIMIT}회 초과")

        for action_payload in pending_actions:
            yield ("action", action_payload)

        full_bot_response = "".join(accumulated_bot_text).strip()
        if full_bot_response:
            await save_turn_history(
                user_uuid, message_text, valid_attachments, recorded_tools, full_bot_response
            )

        yield ("done", {"status": "completed"})

    except UsageLimitExceeded:
        fallback_msg = "요청을 처리하지 못했어요. 조금 더 간단하게 다시 말씀해 주시겠어요?"
        partial = "".join(accumulated_bot_text).strip()
        saved_bot_msg = f"{partial}\n\n{fallback_msg}" if partial else fallback_msg
        try:
            await save_turn_history(
                user_uuid, message_text, valid_attachments, recorded_tools, saved_bot_msg
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
                await save_turn_history(
                    user_uuid, message_text, valid_attachments, recorded_tools, saved_bot_msg
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
                await save_turn_history(
                    user_uuid, message_text, valid_attachments, recorded_tools, saved_bot_msg
                )
            except Exception as save_err:
                logger.error(f"[Redis 에러] 에러 턴 저장 실패 ({user_uuid}): {save_err}")
            yield ("error", {"error": "답변 생성 중 문제가 발생했습니다."})
