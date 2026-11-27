"""백엔드 Gateway 서명 경로(/ingest/members/**) 연동 클라이언트.

v2 API 명세서 9장:
- GET  /ingest/members/{userId}/metrics
- GET  /ingest/members/{userId}/embedding
- PUT  /ingest/members/{userId}/embedding
- 서명 문자열: METHOD\nPATH+QUERY\nTIMESTAMP\nNONCE\nSHA256(BODY)
"""

import hashlib
import hmac
import json
import time
import uuid
from typing import Any

import httpx

from app.config import constant, logger, setting
from app.server.http import get_backend_client

__all__ = [
    "is_ingest_configured",
    "build_ingest_headers",
    "fetch_member_metrics",
    "fetch_member_embedding",
    "put_member_embedding",
]


def is_ingest_configured() -> bool:
    """회원 데이터 서명 키(ingest.member-keys.*) 설정 여부"""
    return bool(setting.INGEST_MEMBER_KEY_ID.strip() and setting.INGEST_MEMBER_SECRET.strip())


def build_ingest_headers(
    method: str,
    path_with_query: str,
    body_bytes: bytes = b"",
    *,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Gateway IngestFilter 규격의 HMAC-SHA256 서명 헤더 생성"""
    ts = timestamp or str(int(time.time()))
    nc = nonce or uuid.uuid4().hex
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    signing_str = f"{method.upper()}\n{path_with_query}\n{ts}\n{nc}\n{body_hash}"
    signature = hmac.new(
        setting.INGEST_MEMBER_SECRET.encode("utf-8"),
        signing_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-Ingest-Key-Id": setting.INGEST_MEMBER_KEY_ID.strip(),
        "X-Ingest-Timestamp": ts,
        "X-Ingest-Nonce": nc,
        "X-Ingest-Signature": signature,
    }
    if setting.BACKEND_HOST.strip():
        headers["Host"] = setting.BACKEND_HOST.strip()
    return headers


async def fetch_member_metrics(user_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    """백엔드에서 회원의 최근 행동 메트릭 조회 (키 미설정/실패 시 빈 목록)"""
    if not is_ingest_configured():
        return []

    path = f"/ingest/members/{user_id}/metrics?limit={limit}"
    url = f"{setting.BACKEND_URL.rstrip('/')}{path}"
    headers = build_ingest_headers("GET", path)

    try:
        res = await get_backend_client().get(
            url,
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
        if res.status_code != 200:
            logger.warning(f"[메트릭 조회 실패] user={user_id} status={res.status_code}")
            return []
        data = res.json()
        if isinstance(data, dict) and data.get("isSuccess") is not False:
            result = data.get("result")
            if isinstance(result, dict):
                return list(result.get("items") or result.get("events") or [])
            if isinstance(result, list):
                return result
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"[메트릭 조회 오류] user={user_id}: {e}")
    return []


async def fetch_member_embedding(user_id: str) -> dict[str, Any] | None:
    """백엔드 DB에 백업된 회원 취향 벡터 조회"""
    if not is_ingest_configured():
        return None

    path = f"/ingest/members/{user_id}/embedding"
    url = f"{setting.BACKEND_URL.rstrip('/')}{path}"
    headers = build_ingest_headers("GET", path)

    try:
        res = await get_backend_client().get(
            url,
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
        if res.status_code != 200:
            return None
        data = res.json()
        if isinstance(data, dict) and data.get("isSuccess") is not False:
            result = data.get("result")
            # 백엔드 TEXT 컬럼 직렬화 형태(dict, JSON 문자열, {"embedding": ...}) 모두 호환
            if isinstance(result, str) and result.strip():
                parsed = json.loads(result)
                return parsed if isinstance(parsed, dict) else None
            if isinstance(result, dict):
                inner = result.get("embedding")
                if isinstance(inner, str) and inner.strip():
                    parsed = json.loads(inner)
                    return parsed if isinstance(parsed, dict) else None
                if isinstance(inner, dict):
                    return inner
                return result
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"[회원 벡터 조회 오류] user={user_id}: {e}")
    return None


async def put_member_embedding(user_id: str, payload: dict[str, Any]) -> bool:
    """회원 취향 벡터를 백엔드 DB로 Write-Back 백업"""
    if not is_ingest_configured():
        return False

    path = f"/ingest/members/{user_id}/embedding"
    url = f"{setting.BACKEND_URL.rstrip('/')}{path}"
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = build_ingest_headers("PUT", path, body_bytes)
    headers["Content-Type"] = "application/json"

    try:
        res = await get_backend_client().put(
            url,
            content=body_bytes,
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
        if res.status_code != 200:
            logger.warning(f"[회원 벡터 백업 실패] user={user_id} status={res.status_code}")
            return False
        data = res.json()
        return not (isinstance(data, dict) and data.get("isSuccess") is False)
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"[회원 벡터 백업 오류] user={user_id}: {e}")
        return False
