import httpx
from fastapi import HTTPException, Request

from app.config import constant, logger, setting

_http_client: httpx.AsyncClient | None = None


async def close() -> None:
    """HTTP 클라이언트 정리"""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def get_user_uuid(request: Request) -> str:
    """
    백엔드 Gateway(G3 GET /api/auth/me) 실시간 호출을 통한 JWT 인증 및 유저 식별자 반환
    """
    global _http_client

    # 1. 토큰 추출 (명세서 AT 쿠키 우선 -> 기존 access_token 쿠키 -> Authorization: Bearer 헤더)
    token = (
        request.cookies.get(constant.AUTH_COOKIE_NAME)
        or request.cookies.get(constant.LEGACY_COOKIE_NAME)
        or None
    )

    if not token:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            candidate = auth_header[7:].strip()
            if candidate and candidate != "cookie_session":
                token = candidate

    # 로컬 개발 환경(ENVIRONMENT=dev)에서 원격 배포 백엔드(https://api.indig.co.kr) 사용 시
    # 브라우저가 .indig.co.kr 전용 HttpOnly 쿠키(AT)를 localhost:8000으로 전송할 수 없으므로,
    # 프론트엔드가 /api/auth/me 검증 후 전달한 X-User-Id 헤더를 개발 모드에 한해 허용
    if not token and not setting.is_production:
        dev_user_id = request.headers.get("x-user-id", "").strip()
        if dev_user_id:
            return dev_user_id

    if not token:
        raise HTTPException(status_code=401, detail="인증 토큰이 필요합니다.")

    # 2. HTTP 클라이언트 준비
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=constant.AUTH_HTTP_TIMEOUT_SECONDS)

    # 3. 백엔드 인증 서버(Gateway /api/auth/me) 실시간 검증 요청
    headers = {
        "Authorization": f"Bearer {token}",
        "Cookie": f"{constant.AUTH_COOKIE_NAME}={token}; {constant.LEGACY_COOKIE_NAME}={token}",
    }

    try:
        res = await _http_client.get(
            constant.AUTH_URL,
            headers=headers,
        )
    except httpx.HTTPError as e:
        logger.error(f"[인증 서버 연결 실패] {constant.AUTH_URL}: {e}")
        raise HTTPException(
            status_code=503, detail="인증 서버에 연결할 수 없습니다."
        ) from e

    # 4. HTTP 상태 코드 검증
    if res.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="유효하지 않거나 만료된 토큰입니다.")
    if res.status_code != 200:
        logger.error(f"[인증 서버 비정상 응답] status={res.status_code}, url={constant.AUTH_URL}")
        raise HTTPException(status_code=502, detail="인증 서버 응답이 올바르지 않습니다.")

    # 5. 응답 JSON 파싱 (명세서 §0-4 봉투 및 §1 G3 규격)
    try:
        data = res.json()
    except Exception as e:
        logger.error(f"[인증 서버 응답 JSON 파싱 실패]: {e}")
        raise HTTPException(status_code=502, detail="인증 서버 응답 형식이 올바르지 않습니다.")

    # 백엔드 명세서 isSuccess 실패 플래그 검증
    if isinstance(data, dict) and data.get("isSuccess") is False:
        err_msg = data.get("message") or "유효하지 않거나 만료된 토큰입니다."
        raise HTTPException(status_code=401, detail=err_msg)

    # 유저 식별자 추출 (명세서 G3: result.userId / 하위 호환: id)
    result = data.get("result", {}) if isinstance(data, dict) else {}
    user_id = (
        (result.get("userId") if isinstance(result, dict) else None)
        or (result.get("id") if isinstance(result, dict) else None)
        or (data.get("userId") if isinstance(data, dict) else None)
        or (data.get("id") if isinstance(data, dict) else None)
    )

    if user_id is None or str(user_id).strip() == "":
        logger.error(f"[인증 서버 응답 오류] 응답에 userId 식별자가 없습니다: {data}")
        raise HTTPException(status_code=502, detail="인증 서버 응답에 유저 정보가 없습니다.")

    return str(user_id)
