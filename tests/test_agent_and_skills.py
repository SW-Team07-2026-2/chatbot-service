import asyncio
from types import SimpleNamespace

from app.chatbot import agent, skill
from app.config import constant


async def test_build_system_instruction_passes_user_uuid(monkeypatch):
    received = {}

    async def fake_get_memories(user_uuid=None, auth=None):
        received["user_uuid"] = user_uuid
        return ["2026년 11월 27일 10시 00분: 검정치마 좋아함"]

    monkeypatch.setattr(agent, "get_user_memories", fake_get_memories)

    prompt = await agent.build_system_instruction("user-42")
    assert received["user_uuid"] == "user-42"
    assert "<user_memory>" in prompt
    assert "검정치마 좋아함" in prompt


async def test_ensure_skills_loaded_and_execute_tool(monkeypatch):
    agent.ensure_skills_loaded()
    assert "open_test_popup" in agent.TOOLS
    assert "control_bgm" in agent.TOOLS
    assert "web_search" in agent.TOOLS
    assert "update_long_term_memory" in agent.TOOLS

    res = await agent.execute_tool("control_bgm", {"bgm_action": "stop"})
    assert res["action"] == "stop_bgm"

    unknown = await agent.execute_tool("non_existent_tool", {})
    assert "error" in unknown


async def test_execute_tool_timeout(monkeypatch):
    async def slow_tool():
        await asyncio.sleep(1.0)
        return "done"

    monkeypatch.setitem(agent.TOOLS, "_slow_test_tool", slow_tool)
    monkeypatch.setattr(constant, "TOOL_TIMEOUT_SECONDS", 0.05)

    res = await agent.execute_tool("_slow_test_tool", {})
    assert "error" in res
    assert "시간이 초과" in res["error"]


async def test_web_search_uses_lazy_exa_client(monkeypatch):
    class FakeExa:
        async def search(self, query, **kwargs):
            return SimpleNamespace(
                output=SimpleNamespace(
                    content="잔나비 새 앨범 발매 [1]",
                    grounding=[SimpleNamespace(confidence="high")],
                )
            )

    monkeypatch.setattr(skill, "get_exa_client", lambda: FakeExa())

    result = await skill.web_search("잔나비 신곡")
    assert result["result"] == "잔나비 새 앨범 발매"
    assert result["confidence"] == "high"


async def test_read_previous_attachment_success_and_missing(monkeypatch):
    monkeypatch.setattr(skill, "get_current_user_uuid", lambda: "u-file")

    async def fake_history(user_uuid: str):
        return [
            {
                "role": "user",
                "content": "포스터 올려요",
                "attachments": [
                    {
                        "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/p1",
                        "file_name": "concert_poster.png",
                        "media_type": "image/png",
                    }
                ],
            }
        ]

    async def fake_generate_content(model, contents, config):
        return SimpleNamespace(text="공연 일시는 12월 5일 오후 7시입니다.")

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    )

    monkeypatch.setattr(skill, "get_raw_chat_history", fake_history)
    monkeypatch.setattr(skill, "get_genai_client", lambda: fake_client)

    res = await skill.read_previous_attachment("공연 시간 알려줘", "concert_poster.png")
    assert res["file_name"] == "concert_poster.png"
    assert "12월 5일" in res["analysis"]

    missing = await skill.read_previous_attachment("내용 알려줘", "없는파일.pdf")
    assert "error" in missing

