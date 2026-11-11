import json
from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.ai.conversation import (
    CHAT_MAX,
    RedisServiceUnavailable,
    get_raw_chat_history,
    raw_history_to_frontend,
    stream_conversation_events,
)
from app.ai.uploader import upload_file_to_gemini
from app.server.auth import get_user_uuid
from app.config import logger

router = APIRouter(tags=["chat"])

UserUUID = Annotated[str, Depends(get_user_uuid)]

class ChatAttachment(BaseModel):
    file_uri: str
    file_name: str | None = None
    media_type: str | None = None
    # 프론트엔드 ChatAttachment 호환용 필드
    url: str | None = None
    name: str | None = None
    type: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(default="", description="사용자 질문 또는 지시 텍스트")
    attachments: list[ChatAttachment] = Field(
        default_factory=list, description="Google Gemini File API에 업로드된 첨부 파일 목록"
    )


#==========================================================================================================

@router.get("/chat")
async def get_chat_history(user_uuid: UserUUID):
    """
    대화 기록 조회, 기록 없으면 첫인사 자동 생성
    """
    try:
        raw_history = await get_raw_chat_history(user_uuid)
    except RedisServiceUnavailable as e:
        logger.error(f"Redis 연결 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="챗봇 시스템 문제 발생.",
        ) from e

    frontend_messages = raw_history_to_frontend(raw_history)
    total_count = len(raw_history)

    return {
        "count": len(frontend_messages),
        "total_count": total_count,
        "has_truncated": total_count >= CHAT_MAX,
        "messages": frontend_messages,
    }


@router.post("/chat/upload")
async def upload_chat_file(
    user_uuid: UserUUID,
    file: UploadFile = File(...),
):
    """
    파일(PDF, 이미지, 텍스트)을 Google Gemini File API에 임시(48시간) 업로드합니다.
    반환된 file_uri를 POST /chat 의 attachments 항목에 담아 질문을 전송하면
    AI가 해당 문서를 읽고 답변합니다.
    """
    file_bytes = await file.read()
    return await upload_file_to_gemini(
        file_bytes=file_bytes,
        filename=file.filename or "uploaded_file",
        content_type=file.content_type,
    )


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request, user_uuid: UserUUID):
    """
    SSE 스트리밍 엔드포인트 (멀티모달 첨부 파일 및 멀티턴 대화 지원)

    - event: status     -> data: {"status": "thinking"|"tool_calling", "message": "...", "tool"?: "..."}
    - event: text_delta -> data: {"content": "..."}
    - event: action     -> data: {"action": "...", ...}
    - event: done       -> data: {"status": "completed"}
    - event: error      -> data: {"error": "...", "code": "..."}
    """
    # 첨부 파일 목록 직렬화 (Pydantic 모델 -> dict)
    attachments_data = [att.model_dump() for att in payload.attachments]

    async def sse_event_generator():
        try:
            async for event_name, event_data in stream_conversation_events(
                user_uuid=user_uuid,
                message=payload.message,
                attachments=attachments_data,
                is_disconnected_check=request.is_disconnected,
            ):
                yield ServerSentEvent(
                    event=event_name,
                    data=json.dumps(event_data, ensure_ascii=False),
                )
        except Exception as e:
            logger.error(f"SSE 스트리밍 전송 에러: {e}")
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"error": "답변 전송 중 문제 발생."}, ensure_ascii=False),
            )

    return EventSourceResponse(sse_event_generator())
