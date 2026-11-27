from types import SimpleNamespace

from google.genai import types

from app.chatbot import conversation
from app.config import constant


def _make_chunk(text: str | None = None, function_call: types.FunctionCall | None = None):
    parts = []
    if text is not None:
        parts.append(SimpleNamespace(text=text, thought=False, function_call=None))
    if function_call is not None:
        parts.append(SimpleNamespace(text=None, thought=False, function_call=function_call))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )


async def test_stream_conversation_success_with_tool_action(monkeypatch):
    saved_calls = []

    async def fake_history(user_uuid: str):
        return []

    async def fake_save(user_uuid, message_text, valid_attachments, recorded_tools, bot_text):
        saved_calls.append((user_uuid, message_text, recorded_tools, bot_text))

    async def fake_sys(user_uuid: str):
        return "system"

    async def fake_exec(name: str, args: dict):
        return {"action": "stop_bgm", "message_for_ai": "BGM 정지됨"}

    call_idx = {"n": 0}

    async def fake_stream(model, contents, config):
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            async def gen1():
                yield _make_chunk(
                    function_call=types.FunctionCall(
                        name="control_bgm", args={"bgm_action": "stop"}
                    )
                )
            return gen1()

        async def gen2():
            yield _make_chunk(text="BGM을 ")
            yield _make_chunk(text="꺼드렸어요.")
        return gen2()

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=fake_stream))
    )

    monkeypatch.setattr(conversation, "get_raw_chat_history", fake_history)
    monkeypatch.setattr(conversation, "save_turn_history", fake_save)
    monkeypatch.setattr(conversation, "build_system_instruction", fake_sys)
    monkeypatch.setattr(conversation, "execute_tool", fake_exec)
    monkeypatch.setattr(conversation, "get_genai_client", lambda: fake_client)

    events = [
        ev
        async for ev in conversation.stream_conversation_events(
            user_uuid="u1", message="노래 꺼줘"
        )
    ]

    event_names = [e[0] for e in events]
    assert "text_delta" in event_names
    assert ("action", {"action": "stop_bgm"}) in events
    assert events[-1] == ("done", {"status": "completed"})
    assert len(saved_calls) == 1
    assert saved_calls[0][3] == "BGM을 꺼드렸어요."


async def test_stream_conversation_errors_do_not_pollute_history(monkeypatch):
    saved_calls = []

    async def fake_history(user_uuid: str):
        return []

    async def fake_save(*args, **kwargs):
        saved_calls.append(args)

    async def fake_sys(user_uuid: str):
        return "system"

    async def failing_stream(model, contents, config):
        raise TimeoutError("request timeout")

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=failing_stream))
    )

    monkeypatch.setattr(conversation, "get_raw_chat_history", fake_history)
    monkeypatch.setattr(conversation, "save_turn_history", fake_save)
    monkeypatch.setattr(conversation, "build_system_instruction", fake_sys)
    monkeypatch.setattr(conversation, "get_genai_client", lambda: fake_client)

    events = [
        ev
        async for ev in conversation.stream_conversation_events(
            user_uuid="u1", message="안녕"
        )
    ]
    assert events[-1] == ("done", {"status": "timeout"})
    assert saved_calls == []


async def test_stream_conversation_usage_limit_does_not_pollute_history(monkeypatch):
    saved_calls = []

    async def fake_history(user_uuid: str):
        return []

    async def fake_save(*args, **kwargs):
        saved_calls.append(args)

    async def fake_sys(user_uuid: str):
        return "system"

    async def fake_exec(name: str, args: dict):
        return {"result": "ok"}

    async def looping_stream(model, contents, config):
        async def gen():
            yield _make_chunk(
                function_call=types.FunctionCall(name="web_search", args={"query": "loop"})
            )
        return gen()

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=looping_stream))
    )

    monkeypatch.setattr(constant, "AGENT_REQUEST_LIMIT", 2)
    monkeypatch.setattr(conversation, "get_raw_chat_history", fake_history)
    monkeypatch.setattr(conversation, "save_turn_history", fake_save)
    monkeypatch.setattr(conversation, "build_system_instruction", fake_sys)
    monkeypatch.setattr(conversation, "execute_tool", fake_exec)
    monkeypatch.setattr(conversation, "get_genai_client", lambda: fake_client)

    events = [
        ev
        async for ev in conversation.stream_conversation_events(
            user_uuid="u1", message="계속 검색해"
        )
    ]
    assert events[-1] == ("done", {"status": "limit_exceeded"})
    assert saved_calls == []
