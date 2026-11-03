from datetime import datetime, timedelta, timezone
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart, TextPartDelta
from pydantic_ai.usage import UsageLimits
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.ai.agent import (
    AGENT_REQUEST_LIMIT,
    SESSION_MAX_MESSAGES,
    SESSION_TTL_SECONDS,
    agent,
)
from app.server.__memory import (
    RedisServiceUnavailable,
    clear_conversation_history,
    get_conversation_history,
    get_conversation_session,
    save_conversation_history,
)
from app.config import logger, settings

router = APIRouter(tags=["Chat"])

# 한 요청에서 모델을 최대 N번까지만 호출한다. (도구 호출 반복이 끝없이 이어지는 것 방지)
USAGE_LIMITS = UsageLimits(request_limit=AGENT_REQUEST_LIMIT)

# 도구(Tool) 호출 시 프론트에 보여줄 커스텀 상태 메시지 매핑 테이블
TOOL_STATUS_MESSAGES: dict[str, str] = {
    "search_long_term_memory": "이전 대화와 음악 취향을 확인하고 있어요... 💭",
    "update_long_term_memory": "새로운 취향 정보를 기억하는 중이에요... 📝",
    "print_test": "서버 콘솔에 테스트 출력 중이에요... 🖨️",
    # 추후 추가될 도구들
    "search_bands": "인디 밴드 정보를 검색하고 있어요... 🎸",
    "get_concert_schedule": "공연 일정을 확인하고 있어요... 🎪",
}

# 챗봇 첫 입장 인사 트리거 키워드
GREETING_TRIGGER = "[INIT_GREETING]"


def build_greeting_prompt() -> str:
    """새로 입장한 유저를 위한 시간대/요일 감성 맞춤 첫인사 프롬프트 생성 (한국 시간 KST 기준)"""
    now = datetime.now(KST)
    hour = now.hour
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    weekday = weekdays[now.weekday()]
    time_desc = "새벽" if hour < 6 else "아침" if hour < 12 else "오후" if hour < 18 else "저녁/밤"

    return (
        f"[시스템 이벤트: 유저가 방금 INDIG 대화방을 열었습니다.]\n"
        f"현재 한국 시각: {weekday}요일 {time_desc}({hour}시)\n"
        f"당신은 INDIG(인디 밴드 통합 관리 서비스)의 AI 가이드 챗봇 '보경이'입니다.\n"
        f"유저에게 시간대나 요일 감성에 어울리는 다정하고 산뜻한 첫인사를 건네세요.\n"
        f"오늘 기분이나 듣고 싶은 음악 장르/분위기, 또는 찾고 있는 인디 공연이 있는지 2~3문장으로 간결하게 물어보세요.\n"
        f"절대 반말을 쓰지 말고 품위 있는 존댓말을 유지하세요."
    )


def sanitize_saved_messages(messages: list, is_greeting: bool) -> list:
    """대화 기록 저장 시, 첫인사 시스템 프롬프트 및 비정상 텍스트 정제"""
    if not messages:
        return messages

    for msg in messages:
        if getattr(msg, "parts", None):
            cleaned_parts = []
            for part in msg.parts:
                part_kind = getattr(part, "part_kind", "")
                # 첫인사 프롬프트가 Redis에 지저분하게 남지 않도록 '(대화 시작)'으로 대체
                if is_greeting and part_kind == "user-prompt" and hasattr(part, "content"):
                    part.content = "(대화 시작)"
                # 혹시 모델이 도구 호출 구문(call:...)을 일반 텍스트로 오출력한 경우 필터링
                if part_kind == "text" and hasattr(part, "content"):
                    if str(part.content).strip().startswith("call:"):
                        continue
                cleaned_parts.append(part)
            msg.parts = cleaned_parts
    return messages


def build_debug_context(user_id: str, message: str, history: list) -> dict:
    """봇(LLM/Agent)에 주입되는 모든 컨텍스트와 메타데이터를 수집한다. (개발용)"""
    # 1. 시스템 프롬프트(지침) 추출
    instructions = ""
    if hasattr(agent, "_instructions"):
        instructions = "\n\n".join(
            str(getattr(i, "instruction", i)) for i in agent._instructions
        ).strip()

    # 2. 등록된 도구(Tools/Skills) 및 파라미터 스키마 추출
    tools_info = []
    function_toolset = getattr(agent, "_function_toolset", None)
    if function_toolset and hasattr(function_toolset, "tools"):
        for name, tool in function_toolset.tools.items():
            param_schema = {}
            if hasattr(tool, "function_schema") and hasattr(tool.function_schema, "json_schema"):
                param_schema = tool.function_schema.json_schema
            tools_info.append({
                "name": name,
                "description": tool.description or "",
                "parameters": param_schema,
            })

    # 3. 모델 설정
    model_settings = agent.model_settings if isinstance(agent.model_settings, dict) else {}

    # 4. 이전 대화 기록 (Redis 단기 기억) 직렬화
    serialized_history = []
    try:
        serialized_history = ModelMessagesTypeAdapter.dump_python(history, mode="json")
    except Exception as e:
        serialized_history = [{"error": f"직렬화 실패: {e}"}]

    return {
        "timestamp": datetime.now().isoformat(),
        "request": {
            "user_id": user_id,
            "message": message,
        },
        "agent": {
            "model": str(settings.LLM_MODEL),
            "instructions": instructions,
            "model_settings": model_settings,
            "retries": getattr(agent, "retries", 3),
            "tool_timeout": getattr(agent, "tool_timeout", 5.0),
            "request_limit": AGENT_REQUEST_LIMIT,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "session_max_messages": SESSION_MAX_MESSAGES,
        },
        "tools": tools_info,
        "history": serialized_history,
        "history_count": len(history),
    }


class ChatRequest(BaseModel):
    message: str
    user_id: str = Field(
        default="guest_user",
        description="유저 식별자 (백엔드 JWT 연동 전까지 테스트용으로 사용, 추후 JWT에서 추출)",
    )


class ChatResponse(BaseModel):
    reply: str


class FrontendChatMessage(BaseModel):
    id: str
    role: str  # "bot" | "user"
    text: str
    time: str | None = None


class ChatHistoryResponse(BaseModel):
    user_id: str
    count: int
    total_count: int = 0
    has_truncated: bool = False
    messages: list[FrontendChatMessage]


KST = timezone(timedelta(hours=9))


def format_korean_time(dt: datetime | None) -> str:
    """UTC 기반 timestamp를 한국 표준시(KST, UTC+9)로 변환하여 '오전/오후 H:MM' 포맷팅"""
    if not dt or not isinstance(dt, datetime):
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_kst = dt.astimezone(KST)
    period = "오후" if dt_kst.hour >= 12 else "오전"
    formatted_hour = dt_kst.hour % 12 or 12
    return f"{period} {formatted_hour}:{dt_kst.minute:02d}"


def format_history_for_frontend(messages: list) -> list[FrontendChatMessage]:
    """Redis의 ModelMessage 목록을 프론트엔드 UI(BotArea)용 ChatMessage 포맷으로 변환"""
    formatted: list[FrontendChatMessage] = []
    idx = 0

    for msg in messages:
        parts = getattr(msg, "parts", [])
        timestamp_str = None

        # 1. 메시지 레벨 timestamp 확인
        msg_ts = getattr(msg, "timestamp", None)
        if msg_ts and isinstance(msg_ts, datetime):
            timestamp_str = format_korean_time(msg_ts)

        # 2. 파트 레벨 timestamp 확인
        if not timestamp_str:
            for part in parts:
                ts = getattr(part, "timestamp", None)
                if ts and isinstance(ts, datetime):
                    timestamp_str = format_korean_time(ts)
                    break

        # 3. 없으면 기본 KST 시간
        if not timestamp_str:
            timestamp_str = format_korean_time(None)

        for part in parts:
            part_kind = getattr(part, "part_kind", "")
            if part_kind == "user-prompt":
                content = str(getattr(part, "content", "")).strip()
                # 첫인사 유도용 시스템 트리거는 유저 말풍선에서 숨김
                if content and content not in ["(대화 시작)", GREETING_TRIGGER, "[INIT]"]:
                    idx += 1
                    formatted.append(
                        FrontendChatMessage(
                            id=f"hist-user-{idx}",
                            role="user",
                            text=content,
                            time=timestamp_str,
                        )
                    )
            elif part_kind == "text":
                content = str(getattr(part, "content", "")).strip()
                if content and not content.startswith("call:"):
                    idx += 1
                    formatted.append(
                        FrontendChatMessage(
                            id=f"hist-bot-{idx}",
                            role="bot",
                            text=content,
                            time=timestamp_str,
                        )
                    )

    return formatted


@router.get(
    "/chat/history",
    response_model=ChatHistoryResponse,
    summary="최근 대화 기록 복원 조회 (새로고침 시 호출)",
)
async def get_history(user_id: str = "guest_user"):
    """
    페이지 새로고침 시 기존 대화 내역을 복원하기 위한 엔드포인트
    - count > 0 이면 기존 대화로 화면 복원
    - count == 0 이면 이력이 없으므로 [INIT]으로 첫인사 생성 요청
    """
    try:
        session = await get_conversation_session(user_id)
    except RedisServiceUnavailable as e:
        logger.error(f"[REDIS 장애] /chat/history 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="세션 저장소(Redis)에 연결할 수 없습니다. 서비스 상태를 확인해 주세요.",
        ) from e

    frontend_messages = format_history_for_frontend(session.messages)
    return ChatHistoryResponse(
        user_id=user_id,
        count=len(frontend_messages),
        total_count=session.total_count,
        has_truncated=session.has_truncated,
        messages=frontend_messages,
    )


@router.delete("/chat/history", summary="유저의 최근 대화 기록 초기화 (세션 리셋)")
async def clear_history(user_id: str = "guest_user"):
    """대화 새로 시작하기 버튼을 누를 때 Redis 세션을 비우는 엔드포인트"""
    try:
        success = await clear_conversation_history(user_id)
        return {"status": "cleared" if success else "failed", "user_id": user_id}
    except RedisServiceUnavailable as e:
        logger.error(f"[REDIS 장애] /chat/history DELETE 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="세션 저장소(Redis)에 연결할 수 없습니다. 서비스 상태를 확인해 주세요.",
        ) from e


@router.post("/chat", response_model=ChatResponse, summary="일반 단발성 대화 응답")
async def chat(body: ChatRequest):
    # 1. Redis에서 이 유저의 최근 대화 세션 불러오기
    try:
        session = await get_conversation_session(body.user_id)
    except RedisServiceUnavailable as e:
        logger.error(f"[REDIS 장애] /chat 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="세션 저장소(Redis)에 연결할 수 없습니다. 서비스 상태를 확인해 주세요.",
        ) from e

    history = session.messages

    is_init = body.message.strip() in ["[INIT]", GREETING_TRIGGER] or not body.message.strip()
    if is_init and len(history) > 0:
        last_bot = next(
            (m for m in reversed(format_history_for_frontend(history)) if m.role == "bot"),
            None,
        )
        return ChatResponse(reply=last_bot.text if last_bot else "이전 대화를 이어가세요!")

    is_greeting = is_init
    prompt = build_greeting_prompt() if is_greeting else body.message

    try:
        # 2. 이전 대화 기록을 넘겨서 에이전트 실행
        result = await agent.run(
            prompt,
            message_history=history,
            usage_limits=USAGE_LIMITS,
        )
    except UsageLimitExceeded:
        # 한도 초과 시 에러로 끊지 않고 유저에게 안내 문구를 돌려준다.
        return ChatResponse(reply="요청을 처리하지 못했어요. 조금 더 간단하게 다시 말씀해 주시겠어요?")

    # 3. 새로운 대화가 추가된 최근 10턴 기록을 Redis에 저장 (TTL 1시간 갱신)
    all_msgs = sanitize_saved_messages(result.all_messages(), is_greeting)
    try:
        await save_conversation_history(
            body.user_id,
            all_msgs,
            prev_total_count=session.total_count,
            prev_history_len=len(history),
        )
    except RedisServiceUnavailable as e:
        logger.error(f"[REDIS 장애] /chat 저장 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="대화는 생성되었으나 세션 저장소(Redis) 연결 실패로 저장되지 않았습니다.",
        ) from e

    return ChatResponse(reply=result.output)


@router.post("/chat/stream", summary="실시간 SSE 스트리밍 대화")
async def chat_stream(body: ChatRequest, request: Request):
    """
    실시간 SSE (Server-Sent Events) 스트리밍 엔드포인트
    - 프론트엔드 노치 UI에서 실시간 타이핑 효과 및 상태 피드백 구현
    - 이벤트 규약:
      - event: history       -> data: {"count": N, "total_count": N, "has_truncated": bool, "messages": [...]}
      - event: debug_context -> data: {...} (봇에게 전달되는 모든 컨텍스트 정보)
      - event: status        -> data: {"status": "thinking"|"tool_calling", "message": "...", "tool"?: "..."}
      - event: text_delta    -> data: {"content": "..."}
      - event: done          -> data: {"status": "completed", "usage": {...}, "all_messages": [...]}
      - event: error         -> data: {"error": "...", "code": "..."}
    """
    # 1. 세션 조회 시도 - 실패 시 바로 SSE error 이벤트 전송
    try:
        session = await get_conversation_session(body.user_id)
    except RedisServiceUnavailable as e:
        logger.error(f"[REDIS 장애 감지] SSE 시작 불가 ({body.user_id}): {e}")

        async def redis_error_stream():
            yield ServerSentEvent(
                event="error",
                data=json.dumps({
                    "error": "세션 저장소(Redis)에 연결할 수 없습니다. 서비스 상태를 확인해 주세요.",
                    "code": "REDIS_UNAVAILABLE",
                }, ensure_ascii=False),
            )

        return EventSourceResponse(redis_error_stream())

    history = session.messages

    is_init_request = body.message.strip() in ["[INIT]", GREETING_TRIGGER] or not body.message.strip()

    # [서버 주도 최적화]
    # 첫 진입/새로고침 초기화 요청([INIT])인데 이미 기존 대화가 남아있는 경우:
    # -> LLM 호출 비용 0원, 지연 시간 0.01초로 즉시 기존 대화 목록을 event: history로 전송하고 완료!
    if is_init_request and len(history) > 0:
        frontend_messages = format_history_for_frontend(history)

        async def history_event_generator():
            yield ServerSentEvent(
                event="history",
                data=json.dumps({
                    "count": len(frontend_messages),
                    "total_count": session.total_count,
                    "has_truncated": session.has_truncated,
                    "messages": [m.model_dump() for m in frontend_messages],
                }, ensure_ascii=False),
            )
            yield ServerSentEvent(
                event="done",
                data=json.dumps({
                    "status": "completed",
                    "action": "history_restored",
                    "total_count": session.total_count,
                    "has_truncated": session.has_truncated,
                }, ensure_ascii=False),
            )

        return EventSourceResponse(history_event_generator())

    # 대화 기록이 없거나(첫 방문) 일반 질문인 경우:
    is_greeting = is_init_request
    prompt = build_greeting_prompt() if is_greeting else body.message

    async def event_generator():
        try:
            # 1. 스트리밍 시작 전, 봇에게 전달되는 모든 정보(컨텍스트, 설정, 도구, 히스토리 등)를 디버그 이벤트로 전송
            display_message = "(첫 입장 인사 요청)" if is_greeting else body.message
            debug_ctx = build_debug_context(body.user_id, display_message, history)
            yield ServerSentEvent(
                event="debug_context",
                data=json.dumps(debug_ctx, ensure_ascii=False),
            )

            # 2. 초기 생각 중 상태 전송
            yield ServerSentEvent(
                event="status",
                data=json.dumps({
                    "status": "thinking",
                    "message": "반가운 인사를 준비하고 있어요... ✨" if is_greeting else "답변을 생각하고 있어요...",
                }, ensure_ascii=False),
            )

            result_event: AgentRunResultEvent | None = None

            async with agent.run_stream_events(
                prompt,
                message_history=history,
                usage_limits=USAGE_LIMITS,
            ) as events:
                async for event in events:
                    # 클라이언트 연결 끊김 감지 시 루프 중단
                    if await request.is_disconnected():
                        logger.warning(f"클라이언트 연결 끊김 감지 ({body.user_id})")
                        break

                    # A. 도구 호출 시작 (예: search_long_term_memory 등)
                    if isinstance(event, FunctionToolCallEvent):
                        tool_name = getattr(event.part, "tool_name", "")
                        status_msg = TOOL_STATUS_MESSAGES.get(
                            tool_name, f"'{tool_name}' 작업 진행 중..."
                        )
                        yield ServerSentEvent(
                            event="status",
                            data=json.dumps({
                                "status": "tool_calling",
                                "tool": tool_name,
                                "message": status_msg,
                            }, ensure_ascii=False),
                        )

                    # B. 도구 실행 완료 후 다시 생각/정리 중
                    elif isinstance(event, FunctionToolResultEvent):
                        tool_name = getattr(event.part, "tool_name", "")
                        yield ServerSentEvent(
                            event="status",
                            data=json.dumps({
                                "status": "thinking",
                                "tool": tool_name,
                                "message": "결과를 바탕으로 답변을 정리하고 있어요...",
                            }, ensure_ascii=False),
                        )

                    # C. 텍스트 시작 (첫 청크)
                    elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                        if event.part.content and not str(event.part.content).strip().startswith("call:"):
                            yield ServerSentEvent(
                                event="text_delta",
                                data=json.dumps({"content": event.part.content}, ensure_ascii=False),
                            )

                    # D. 텍스트 델타 (후속 청크들)
                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        if event.delta.content_delta and not str(event.delta.content_delta).strip().startswith("call:"):
                            yield ServerSentEvent(
                                event="text_delta",
                                data=json.dumps({"content": event.delta.content_delta}, ensure_ascii=False),
                            )

                    # E. 최종 실행 완료 결과 이벤트
                    elif isinstance(event, AgentRunResultEvent):
                        result_event = event

            # 루프 정상 완료 후 대화 기록 및 usage 처리
            if result_event is not None:
                all_msgs = sanitize_saved_messages(result_event.result.all_messages(), is_greeting)
                updated_session = await save_conversation_history(
                    body.user_id,
                    all_msgs,
                    prev_total_count=session.total_count,
                    prev_history_len=len(history),
                )

                # 토큰 사용량 정보 추출
                usage_info = {}
                try:
                    u = result_event.result.usage
                    usage_info = {
                        "requests": getattr(u, "requests", 1),
                        "request_tokens": getattr(u, "request_tokens", None),
                        "response_tokens": getattr(u, "response_tokens", None),
                        "total_tokens": getattr(u, "total_tokens", None),
                    }
                except Exception:
                    pass

                # 갱신된 전체 메시지 직렬화
                updated_history = []
                try:
                    updated_history = ModelMessagesTypeAdapter.dump_python(all_msgs, mode="json")
                except Exception:
                    pass

                yield ServerSentEvent(
                    event="done",
                    data=json.dumps({
                        "status": "completed",
                        "total_count": updated_session.total_count,
                        "has_truncated": updated_session.has_truncated,
                        "usage": usage_info,
                        "all_messages": updated_history,
                    }, ensure_ascii=False),
                )

        except UsageLimitExceeded:
            yield ServerSentEvent(
                event="text_delta",
                data=json.dumps(
                    {"content": "요청을 처리하지 못했어요. 조금 더 간단하게 다시 말씀해 주시겠어요?"},
                    ensure_ascii=False,
                ),
            )
            yield ServerSentEvent(
                event="done",
                data=json.dumps({"status": "limit_exceeded"}, ensure_ascii=False),
            )
        except RedisServiceUnavailable as e:
            logger.error(f"[REDIS 장애 발생] 스트리밍 중 Redis 저장 실패: {e}")
            yield ServerSentEvent(
                event="error",
                data=json.dumps({
                    "error": "세션 저장소(Redis) 연결이 불안정합니다. 대화 기록 저장에 실패했습니다.",
                    "code": "REDIS_UNAVAILABLE",
                }, ensure_ascii=False),
            )
        except Exception as e:
            logger.error(f"스트리밍 처리 중 에러: {e}")
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": "답변 생성 중 문제가 발생했습니다."}, ensure_ascii=False),
            )

    return EventSourceResponse(event_generator())


@router.get(
    "/chat/debug_context",
    summary="[DEV] 현재 유저의 봇 주입 정보(컨텍스트, 히스토리, 툴) 사전 조회",
    include_in_schema=not settings.is_production,
)
async def get_current_debug_context(user_id: str = "guest_user"):
    if settings.is_production:
        raise HTTPException(status_code=404, detail="Not Found")
    history = await get_conversation_history(user_id)
    return build_debug_context(user_id, message="(대화 입력 전 대기 중)", history=history)

