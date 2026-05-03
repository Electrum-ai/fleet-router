"""Tests for fleet.proxy — Anthropic Messages API compatibility layer."""
from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from fleet.proxy import (
    _flatten_content,
    _maybe_enrich_with_ollama_hint,
    _parse_openai_chat_request,
    _parse_request,
    build_app,
)
from fleet.router import ERROR_ALL_MODELS_FAILED, ERROR_NO_MODEL


class _StubRouter:
    """Stand-in for FleetRouter.ask — captures the prompt + system it saw."""

    def __init__(self, answer: str = "hello back"):
        self._answer = answer
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
        self.last_prompt = prompt
        self.last_system = system
        return self._answer


# ---------- pure unit tests (no HTTP) ----------

def test_flatten_content_string_passthrough():
    assert _flatten_content("hi") == "hi"


def test_flatten_content_text_blocks():
    blocks = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    assert _flatten_content(blocks) == "one\ntwo"


def test_flatten_content_tool_use_summarized():
    blocks = [{"type": "tool_use", "name": "Read", "input": {"path": "/tmp/x"}}]
    out = _flatten_content(blocks)
    assert "tool_call" in out and "Read" in out and "/tmp/x" in out


def test_flatten_content_tool_result_recurses():
    blocks = [{
        "type": "tool_result",
        "tool_use_id": "tool_123",
        "content": [{"type": "text", "text": "result body"}],
    }]
    out = _flatten_content(blocks)
    assert "tool_result" in out and "tool_123" in out and "result body" in out


def test_parse_request_collapses_history_with_role_markers():
    body = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        "max_tokens": 100,
    }
    parsed = _parse_request(body)
    assert "Human: first question" in parsed.prompt
    assert "Assistant: first answer" in parsed.prompt
    assert "Human: second question" in parsed.prompt
    assert parsed.prompt.rstrip().endswith("Assistant:")
    assert parsed.requested_model == "claude-3-5-sonnet"
    assert parsed.stream is False


def test_parse_request_carries_system_string():
    body = {
        "model": "x",
        "system": "you are concise",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _parse_request(body).system == "you are concise"


def test_parse_request_flattens_system_blocks():
    body = {
        "model": "x",
        "system": [{"type": "text", "text": "rule one"}, {"type": "text", "text": "rule two"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _parse_request(body).system == "rule one\nrule two"


def test_parse_request_rejects_empty_messages():
    with pytest.raises(web.HTTPBadRequest):
        _parse_request({"model": "x", "messages": []})


# ---------- HTTP integration tests ----------

@pytest.mark.asyncio
async def test_messages_non_streaming_shape():
    router = _StubRouter("the answer is 42")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "what is the answer"}],
            "max_tokens": 100,
        })
        assert resp.status == 200
        body = await resp.json()

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-3-5-sonnet"
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "the answer is 42"}]
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    assert body["id"].startswith("msg_")
    assert "Human: what is the answer" in (router.last_prompt or "")


@pytest.mark.asyncio
async def test_messages_streaming_event_sequence():
    router = _StubRouter("streamed text payload")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 100,
            "stream": True,
        })
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        raw = (await resp.read()).decode("utf-8")

    # Parse the SSE event sequence.
    events = [
        line[len("event: "):].strip()
        for line in raw.splitlines() if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert events[1] == "content_block_start"
    assert "content_block_delta" in events
    assert events[-3] == "content_block_stop"
    assert events[-2] == "message_delta"
    assert events[-1] == "message_stop"

    # Reassemble the streamed text from data: lines.
    deltas = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: "):])
            if payload.get("type") == "content_block_delta":
                deltas.append(payload["delta"]["text"])
    assert "".join(deltas) == "streamed text payload"


@pytest.mark.asyncio
async def test_api_key_required_when_configured():
    router = _StubRouter()
    app = build_app(router, api_key="secret-token")  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        # No header — rejected.
        bad = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert bad.status == 401

        # Correct header — accepted.
        ok = await client.post(
            "/v1/messages",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "secret-token"},
        )
        assert ok.status == 200


@pytest.mark.asyncio
async def test_healthz():
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True


@pytest.mark.asyncio
async def test_router_failure_returns_500():
    class _BoomRouter:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("ollama unreachable")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 500


# ---------- Ollama-down hint enrichment ----------

def test_maybe_enrich_passes_through_normal_text():
    assert _maybe_enrich_with_ollama_hint("hello world") == "hello world"


def test_maybe_enrich_passes_through_unrelated_parenthesised():
    # Strings that start with "(" but aren't router sentinels stay untouched.
    assert _maybe_enrich_with_ollama_hint("(this is just a parenthetical)") == \
        "(this is just a parenthetical)"


def test_maybe_enrich_appends_hint_to_all_models_failed_sentinel():
    out = _maybe_enrich_with_ollama_hint(ERROR_ALL_MODELS_FAILED)
    assert out.startswith(ERROR_ALL_MODELS_FAILED)
    assert "ollama serve" in out


def test_maybe_enrich_appends_hint_to_no_model_sentinel_with_suffix():
    text = f"{ERROR_NO_MODEL} for tag: code"
    out = _maybe_enrich_with_ollama_hint(text)
    assert out.startswith(text)
    assert "ollama serve" in out


@pytest.mark.asyncio
async def test_messages_enriches_ollama_down_response():
    """When router.ask returns an error sentinel (Ollama down), the proxy
    must return 200 with the sentinel + actionable troubleshooting text —
    NOT a raw HTTP error, since errors mid-stream can't be recovered cleanly."""
    router = _StubRouter(ERROR_ALL_MODELS_FAILED)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 200
        body = await resp.json()
    text = body["content"][0]["text"]
    assert text.startswith(ERROR_ALL_MODELS_FAILED)
    assert "ollama serve" in text


@pytest.mark.asyncio
async def test_messages_streaming_enriches_ollama_down_response():
    router = _StubRouter(ERROR_ALL_MODELS_FAILED)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status == 200
        raw = (await resp.read()).decode("utf-8")
    deltas = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: "):])
            if payload.get("type") == "content_block_delta":
                deltas.append(payload["delta"]["text"])
    full_text = "".join(deltas)
    assert ERROR_ALL_MODELS_FAILED in full_text
    assert "ollama serve" in full_text


# ---------- /v1/models endpoint ----------

@pytest.mark.asyncio
async def test_v1_models_returns_openai_shape():
    """Stub router has no _registry attribute — endpoint must degrade gracefully."""
    router = _StubRouter()
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/models")
        assert resp.status == 200
        body = await resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    # No _registry on stub → empty list, not a 500.
    assert body["data"] == []


# ---------- OpenAI Chat Completions parser ----------

def test_parse_openai_extracts_system_messages_and_collapses_turns():
    body = {
        "model": "fleet-router",
        "messages": [
            {"role": "system", "content": "you are concise"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "say more"},
        ],
    }
    parsed = _parse_openai_chat_request(body)
    assert parsed.system == "you are concise"
    assert "Human: hi" in parsed.prompt
    assert "Assistant: hello" in parsed.prompt
    assert "Human: say more" in parsed.prompt
    assert parsed.prompt.rstrip().endswith("Assistant:")
    assert parsed.requested_model == "fleet-router"
    assert parsed.stream is False


def test_parse_openai_concatenates_multiple_system_messages():
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "rule one"},
            {"role": "system", "content": "rule two"},
            {"role": "user", "content": "ok"},
        ],
    }
    parsed = _parse_openai_chat_request(body)
    assert parsed.system == "rule one\n\nrule two"


def test_parse_openai_summarizes_assistant_tool_calls():
    body = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "what's the weather"},
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "72F sunny"},
            {"role": "user", "content": "thanks"},
        ],
    }
    parsed = _parse_openai_chat_request(body)
    # Tool call surfaced as text inside assistant turn
    assert "[tool_call name=get_weather" in parsed.prompt
    assert "NYC" in parsed.prompt
    # Tool result surfaced as a Human turn (since fleet has no tool role)
    assert "[tool_result id=call_abc]" in parsed.prompt
    assert "72F sunny" in parsed.prompt


def test_parse_openai_rejects_empty_messages():
    with pytest.raises(web.HTTPBadRequest):
        _parse_openai_chat_request({"model": "x", "messages": []})


def test_parse_openai_max_tokens_falls_back_to_max_completion_tokens():
    """OpenAI's newer API renamed max_tokens → max_completion_tokens."""
    parsed = _parse_openai_chat_request({
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 200,
    })
    assert parsed.max_tokens == 200


# ---------- /v1/chat/completions HTTP integration ----------

@pytest.mark.asyncio
async def test_chat_completions_non_streaming_shape():
    router = _StubRouter("the answer is 42")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "fleet-router",
            "messages": [{"role": "user", "content": "what is the answer"}],
        })
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "fleet-router"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "the answer is 42"}
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_chat_completions_streaming_chunk_sequence():
    router = _StubRouter("streamed payload")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        })
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        raw = (await resp.read()).decode("utf-8")

    # Final terminator
    assert raw.rstrip().endswith("data: [DONE]")

    # Walk the data: lines, ignoring SSE comment heartbeats (lines starting with ':')
    chunks: list[dict] = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))

    # First chunk: role marker, no finish_reason
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    assert chunks[0]["object"] == "chat.completion.chunk"

    # Last data chunk: empty delta + finish_reason=stop
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    # Middle chunks reassemble to the full text
    body = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks[1:-1]
    )
    assert body == "streamed payload"


@pytest.mark.asyncio
async def test_chat_completions_api_key_via_authorization_bearer():
    router = _StubRouter()
    app = build_app(router, api_key="secret-token")  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        # Missing -> 401
        bad = await client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
        })
        assert bad.status == 401

        # Bearer token -> OK (the OpenAI SDK default)
        ok = await client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert ok.status == 200


@pytest.mark.asyncio
async def test_chat_completions_enriches_ollama_down_in_streaming():
    router = _StubRouter(ERROR_ALL_MODELS_FAILED)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status == 200
        raw = (await resp.read()).decode("utf-8")
    body = "".join(
        json.loads(line[len("data: "):])["choices"][0]["delta"].get("content", "")
        for line in raw.splitlines()
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    )
    assert ERROR_ALL_MODELS_FAILED in body
    assert "ollama serve" in body


@pytest.mark.asyncio
async def test_v1_models_returns_registry_models():
    class _RegistryStub:
        def all_available(self):
            return ["deepseek-v4-pro", "glm-5.1"]

    class _RouterWithRegistry:
        def __init__(self):
            self._registry = _RegistryStub()

        async def ask(self, *args, **kwargs):
            return "n/a"

    app = build_app(_RouterWithRegistry())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/v1/models")
        body = await resp.json()
    ids = [m["id"] for m in body["data"]]
    assert ids == ["deepseek-v4-pro", "glm-5.1"]
    assert all(m["object"] == "model" and m["owned_by"] == "fleet" for m in body["data"])
