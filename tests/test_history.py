import asyncio
import json

import pytest

from app.chatbot import history
from app.config import constant
from tests.conftest import FakeRedis


async def test_append_and_stable_ids_after_ltrim(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(history, "get_client", lambda: fake_redis)
    monkeypatch.setattr(constant, "CHAT_MAX", 3)

    await history.append_chat_message("u1", "user", "첫 번째")
    await history.append_chat_message("u1", "bot", "두 번째")
    await history.append_chat_message("u1", "user", "세 번째")

    before = history.raw_history_to_frontend(await history.get_raw_chat_history("u1"))
    assert len(before) == 3
    second_id = before[1]["id"]
    third_id = before[2]["id"]
    assert second_id.startswith("msg-")

    # 4번째 메시지 추가로 LTRIM 발생 후에도 기존 메시지 ID가 유지되어야 함
    await history.append_chat_message("u1", "bot", "네 번째")
    after = history.raw_history_to_frontend(await history.get_raw_chat_history("u1"))
    assert len(after) == 3
    assert after[0]["id"] == second_id
    assert after[1]["id"] == third_id


def test_raw_history_to_frontend_legacy_id_fallback():
    raw = [
        {"role": "bot", "content": "안녕", "ts": 1700000000.0},
        {"role": "tool", "tool": "web_search", "content": "ignored", "ts": 1700000001.0},
        {"role": "user", "content": "질문", "ts": 1700000002.0},
    ]
    msgs = history.raw_history_to_frontend(raw)
    assert len(msgs) == 2
    assert msgs[0]["id"] == "hist-bot-1"
    assert msgs[1]["id"] == "hist-user-3"


async def test_initial_greeting_created_once_under_concurrency(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(history, "get_client", lambda: fake_redis)

    # 동시에 5개의 초기 조회가 들어와도 첫인사는 1건만 저장되어야 함
    results = await asyncio.gather(
        *(history.get_raw_chat_history("u-concurrent") for _ in range(5))
    )
    key = history.make_chat_key("u-concurrent")
    stored = [json.loads(x) for x in fake_redis.lists[key]]
    assert len(stored) == 1
    assert stored[0]["role"] == "bot"
    assert all(len(r) == 1 for r in results)


def test_raw_history_to_contents_trims_edges_and_restores_parts():
    raw = [
        {"role": "bot", "content": "초기 인사", "ts": 1700000000.0},
        {
            "role": "user",
            "content": "포스터 봐줘",
            "ts": 1700000010.0,
            "attachments": [
                {
                    "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                    "media_type": "image/png",
                }
            ],
        },
        {
            "role": "tool",
            "tool": "web_search",
            "args": {"query": "인디 공연"},
            "content": {"result": "홍대 공연"},
            "ts": 1700000011.0,
        },
        {"role": "bot", "content": "홍대 공연입니다.", "ts": 1700000012.0},
        {"role": "user", "content": "미완성 마지막 질문", "ts": 1700000020.0},
    ]

    contents = history.raw_history_to_contents(raw)
    # 앞쪽 초기 인사(model)와 뒤쪽 미완성 user는 제거됨
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    assert contents[0].parts[1].file_data.file_uri.endswith("/abc")
    assert contents[-1].role == "model"
    assert contents[-1].parts[0].text == "홍대 공연입니다."


def test_raw_history_to_contents_limits_old_attachments_to_text_placeholder(monkeypatch):
    monkeypatch.setattr(constant, "HISTORY_ATTACHMENT_TURNS", 1)
    raw = [
        {
            "role": "user",
            "content": "옛날 포스터",
            "ts": 1700000001.0,
            "attachments": [
                {
                    "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/old1",
                    "file_name": "old_poster.png",
                    "media_type": "image/png",
                }
            ],
        },
        {"role": "bot", "content": "확인했어요.", "ts": 1700000002.0},
        {
            "role": "user",
            "content": "최근 포스터",
            "ts": 1700000010.0,
            "attachments": [
                {
                    "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/new1",
                    "file_name": "new_poster.png",
                    "media_type": "image/png",
                }
            ],
        },
        {"role": "bot", "content": "최근 포스터도 확인했어요.", "ts": 1700000011.0},
    ]

    contents = history.raw_history_to_contents(raw)
    old_user = contents[0]
    recent_user = contents[2]

    # 오래된 턴: FileData 제거되고 파일명 텍스트만 남음
    assert len(old_user.parts) == 1
    assert old_user.parts[0].file_data is None
    assert "[이전 첨부파일: old_poster.png]" in old_user.parts[0].text

    # 최근 턴: FileData 유지
    assert len(recent_user.parts) == 2
    assert recent_user.parts[1].file_data.file_uri.endswith("/new1")
