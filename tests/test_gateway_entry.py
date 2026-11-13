import os

# app.config 가 import 시점에 환경 변수를 읽는다
os.environ.update(
    ENVIRONMENT="prod",
    REDIS_URL="redis://localhost:6379/0",
    CORS_ORIGINS="http://localhost:5173",
    BACKEND_URL="http://localhost:8080",
    LLM_MODEL="test",
    GEMINI_API_KEY="test",
    EXA_API_KEY="test",
)

from fastapi.testclient import TestClient  # noqa: E402

from gateway_entry import app  # noqa: E402

client = TestClient(app)


def test_chat_routes_are_served_under_gateway_prefix():
    # 인증 없이 부르면 기존 핸들러의 401 — 경로가 없으면 404 다
    for prefix in ("", "/api/chatbot"):
        assert client.get(prefix + "/chat").status_code == 401
        assert client.post(prefix + "/chat", json={"message": "hi"}).status_code == 401
        assert client.post(prefix + "/chat/upload", files={"file": ("a.txt", b"a")}).status_code == 401
    assert client.get("/api/chatbot").status_code == 404


def test_liveness_stays_at_root():
    assert client.get("/health/liveness").json() == {"status": "UP"}
