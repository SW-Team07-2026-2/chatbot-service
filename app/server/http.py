import httpx

from app.config import constant

__all__ = ["get_backend_client", "close"]

_http_client: httpx.AsyncClient | None = None


def get_backend_client() -> httpx.AsyncClient:
    """메인 백엔드(Gateway) 호출용 공유 비동기 HTTP 클라이언트"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS)
    return _http_client


async def close() -> None:
    """HTTP 클라이언트 커넥션 풀 정리"""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
