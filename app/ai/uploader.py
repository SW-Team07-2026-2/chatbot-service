import io
import mimetypes
from pathlib import Path
from fastapi import HTTPException
from google import genai
from google.genai import types

from app.config import logger, setting

# 최대 파일 업로드 크기
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

# MIME 타입 프리픽스
ALLOWED_MIME_PREFIXES = ("image/", "text/", "audio/", "video/")

# MIME 타입 매칭
ALLOWED_EXACT_MIMES = {
    # 문서
    "application/pdf",
    "application/rtf",
    "application/xml",
    # 데이터 코드
    "application/json",
    "application/javascript",
    "application/x-javascript",
    "application/x-python-code",
    "application/x-sh",
    "text/csv",
    "text/markdown",
    "text/tab-separated-values",
}

# 확장자 기반 fallback 매핑
EXTENSION_MIME_MAP = {
    # 문서 데이터
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".xml": "application/xml",
    ".rtf": "application/rtf",
    # 소스 코드
    ".py": "text/x-python",
    ".js": "application/javascript",
    ".ts": "text/plain",
    ".html": "text/html",
    ".css": "text/css",
    ".sh": "application/x-sh",
    # 이미지
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    # 오디오
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/m4a",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    # 비디오
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/avi",
}

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=setting.GEMINI_API_KEY)
    return _client


def validate_file_type(filename: str, content_type: str | None) -> str:
    """MIME 타입 유효성 검사 및 정규화"""
    ext = Path(filename).suffix.lower()
    fallback_mime = EXTENSION_MIME_MAP.get(ext)

    if (not content_type or content_type == "application/octet-stream") and fallback_mime:
        mime = fallback_mime
    else:
        mime = (
            content_type
            or fallback_mime
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

    is_allowed = (
        mime in ALLOWED_EXACT_MIMES
        or any(mime.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES)
    )

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다 ({mime}). 이미지, 문서(PDF/텍스트), 오디오, 비디오 파일만 지원됩니다.",
        )
    return mime


async def upload_file_to_gemini(
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
) -> dict:
    """
    Google Gemini File API 파일 비동기 업로드
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB까지 업로드 가능합니다.",
        )

    mime_type = validate_file_type(filename, content_type)

    try:
        client = get_genai_client()
        stream = io.BytesIO(file_bytes)

        upload_config = types.UploadFileConfig(
            display_name=filename,
            mime_type=mime_type,
        )

        uploaded = await client.aio.files.upload(file=stream, config=upload_config)

        logger.info(f"Gemini File API 업로드 성공: {filename} ({mime_type}) -> {uploaded.uri}")

        return {
            "file_uri": uploaded.uri,
            "file_name": uploaded.display_name or filename,
            "media_type": uploaded.mime_type or mime_type,
            # 프론트엔드 ChatAttachment 호환용 필드
            "url": uploaded.uri,
            "name": uploaded.display_name or filename,
            "type": uploaded.mime_type or mime_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini File API 업로드 실패 ({filename}): {e}")
        raise HTTPException(
            status_code=502,
            detail="파일 업로드 처리 중 외부 AI 서비스 오류가 발생했습니다.",
        ) from e
