import pytest
from fastapi.testclient import TestClient

from app.chatbot import api as chatbot_api
from app.chatbot import history
from app.config import constant
from app.main import app
from app.recommender import store as rec_store
from app.server import api as server_api
from app.server import redis as server_redis
from tests.conftest import FakeRedis


@pytest.fixture(autouse=True)
def _mock_lifespan_redis(monkeypatch):
    fake_redis = FakeRedis()

    async def noop_ping():
        return True

    async def noop_close():
        return None

    monkeypatch.setattr(server_redis, "ping", noop_ping)
    monkeypatch.setattr(server_redis, "close", noop_close)
    monkeypatch.setattr(rec_store, "get_client", lambda: fake_redis)
    monkeypatch.setattr(constant, "REC_AUTO_SYNC_ON_STARTUP", False)


def test_health_endpoints(monkeypatch):
    async def redis_ok():
        return True

    async def redis_down():
        return False

    monkeypatch.setattr(server_api, "ping", redis_ok)
    with TestClient(app) as client:
        res = client.get("/health/liveness")
        assert res.status_code == 200
        assert res.json() == {"status": "UP"}

        res = client.get("/health/readiness")
        assert res.status_code == 200
        assert res.json()["components"]["redis"] == "UP"

        monkeypatch.setattr(server_api, "ping", redis_down)
        res = client.get("/health/readiness")
        assert res.status_code == 503
        assert res.json()["status"] == "DOWN"


def test_chat_requires_auth():
    with TestClient(app) as client:
        res = client.get("/api/chatbot/chat")
        assert res.status_code == 401


def test_get_chat_history_with_dev_header(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(history, "get_client", lambda: fake_redis)

    with TestClient(app) as client:
        res = client.get("/api/chatbot/chat", headers={"X-User-Id": "dev-user-1"})
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 1
        assert body["messages"][0]["role"] == "bot"
        assert body["messages"][0]["id"].startswith("msg-")


def test_chat_rejects_disallowed_file_uri():
    with TestClient(app) as client:
        res = client.post(
            "/api/chatbot/chat",
            headers={"X-User-Id": "dev-user-1"},
            json={
                "message": "봐줘",
                "attachments": [{"file_uri": "https://evil.example.com/file.png"}],
            },
        )
        assert res.status_code == 422


def test_upload_chat_file_validates_type_and_calls_uploader(monkeypatch):
    async def fake_upload(file_bytes: bytes, filename: str, content_type: str | None = None):
        return {
            "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/123",
            "file_name": filename,
            "media_type": content_type or "image/png",
            "url": "https://generativelanguage.googleapis.com/v1beta/files/123",
            "name": filename,
            "type": content_type or "image/png",
        }

    monkeypatch.setattr(chatbot_api, "upload_file_to_gemini", fake_upload)

    with TestClient(app) as client:
        res = client.post(
            "/api/chatbot/chat/upload",
            headers={"X-User-Id": "dev-user-1"},
            files={"file": ("poster.png", b"fake-png-bytes", "image/png")},
        )
        assert res.status_code == 200
        assert res.json()["file_name"] == "poster.png"
