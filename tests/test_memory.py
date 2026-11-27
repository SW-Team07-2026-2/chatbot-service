from types import SimpleNamespace

from app.chatbot import memory
from tests.conftest import FakeRedis


class _FakeHttpResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


async def test_get_user_memories_caches_backend_response(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory, "get_client", lambda: fake_redis)

    calls = {"get": 0}

    class FakeBackendClient:
        async def get(self, url, headers=None, timeout=None):
            calls["get"] += 1
            return _FakeHttpResponse(
                200,
                {
                    "isSuccess": True,
                    "result": {
                        "messages": [
                            {"content": "실리카겔 좋아함", "spokenAt": "2026-11-27T10:00:00+09:00"}
                        ]
                    },
                },
            )

    monkeypatch.setattr(memory, "get_backend_client", lambda: FakeBackendClient())

    # 1차 호출: 캐시 미스 -> 백엔드 조회 후 Redis ZSET 적재
    first = await memory.get_user_memories(user_uuid="u-ltm", auth=("AT=tok", "csrf"))
    assert calls["get"] == 1
    assert len(first) == 1
    assert "실리카겔 좋아함" in first[0]

    # 2차 호출: 캐시 히트 -> 백엔드 추가 호출 없음
    second = await memory.get_user_memories(user_uuid="u-ltm", auth=("AT=tok", "csrf"))
    assert calls["get"] == 1
    assert second == first


async def test_add_user_memory_writes_redis_only_on_backend_success(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory, "get_client", lambda: fake_redis)

    captured = {}

    class FakeBackendClient:
        def __init__(self, ok: bool):
            self.ok = ok

        async def post(self, url, json=None, headers=None, timeout=None):
            captured["headers"] = headers
            captured["json"] = json
            if self.ok:
                return _FakeHttpResponse(200, {"isSuccess": True})
            return _FakeHttpResponse(500, {"isSuccess": False})

    # 실패 시 Redis에 저장되지 않아야 함
    monkeypatch.setattr(memory, "get_backend_client", lambda: FakeBackendClient(ok=False))
    ok = await memory.add_user_memory("홍대 자주 감", user_uuid="u-save", auth=("AT=1", "xsrf-1"))
    assert ok is False
    assert memory._make_ltm_key("u-save") not in fake_redis.zsets

    # 성공 시 CSRF 헤더 포함 전달 & Redis ZSET 반영
    monkeypatch.setattr(memory, "get_backend_client", lambda: FakeBackendClient(ok=True))
    ok = await memory.add_user_memory("홍대 자주 감", user_uuid="u-save", auth=("AT=1", "xsrf-1"))
    assert ok is True
    assert captured["headers"]["Cookie"] == "AT=1"
    assert captured["headers"]["X-XSRF-TOKEN"] == "xsrf-1"
    cached = await memory.get_user_memories(user_uuid="u-save")
    assert any("홍대 자주 감" in item for item in cached)
