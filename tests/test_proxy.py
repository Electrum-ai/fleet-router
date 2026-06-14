"""Tests for fleet.proxy — Anthropic Messages API compatibility layer."""
from __future__ import annotations

import asyncio
import json
import re

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from fleet.proxy import (
    _client_safe_error,
    _coerce_int,
    _default_allowed_hosts,
    _flatten_content,
    _host_only,
    _make_host_guard,
    _maybe_enrich_with_ollama_hint,
    _parse_openai_chat_request,
    _parse_request,
    _resolve_force_model,
    build_app,
)
from fleet.router import ERROR_ALL_MODELS_FAILED, ERROR_NO_MODEL


class _StubRouter:
    """Stand-in for FleetRouter.ask — captures the prompt + system it saw."""

    def __init__(self, answer: str = "hello back", delay: float = 0.0):
        self._answer = answer
        self._delay = delay
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.call_count = 0

    async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
        self.last_prompt = prompt
        self.last_system = system
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
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


# ---------- Force-model resolver ----------

class _FakeRegistry:
    def __init__(self, names):
        self._names = list(names)

    def all_available(self):
        return self._names


class _RouterWithModels:
    def __init__(self, names):
        self._registry = _FakeRegistry(names)

    async def ask(self, *args, **kwargs):
        return "n/a"


def test_resolve_force_model_returns_none_for_unknown_name():
    router = _RouterWithModels(["glm-5.1", "deepseek-v4-pro"])
    # Default placeholder fleet sends when client didn't ask for anything.
    assert _resolve_force_model("fleet-router", router) is None
    # Claude Code sends Anthropic model names — fleet doesn't know them.
    assert _resolve_force_model("claude-opus-4-7", router) is None
    # Empty string short-circuits.
    assert _resolve_force_model("", router) is None


def test_resolve_force_model_returns_known_bare_name():
    router = _RouterWithModels(["glm-5.1", "deepseek-v4-pro"])
    assert _resolve_force_model("glm-5.1", router) == "glm-5.1"
    assert _resolve_force_model("deepseek-v4-pro", router) == "deepseek-v4-pro"


def test_resolve_force_model_strips_provider_prefixes():
    """aider/litellm prefix model names with `openai/` etc."""
    router = _RouterWithModels(["glm-5.1", "kimi-k2.7"])
    assert _resolve_force_model("openai/glm-5.1", router) == "glm-5.1"
    assert _resolve_force_model("anthropic/glm-5.1", router) == "glm-5.1"
    assert _resolve_force_model("ollama/kimi-k2.7", router) == "kimi-k2.7"
    assert _resolve_force_model("fleet/kimi-k2.7", router) == "kimi-k2.7"


def test_resolve_force_model_tolerates_cloud_suffix():
    """Some clients echo back the api_model with `:cloud` — strip and match."""
    router = _RouterWithModels(["deepseek-v4-flash"])
    assert _resolve_force_model("deepseek-v4-flash:cloud", router) == "deepseek-v4-flash"
    assert _resolve_force_model("openai/deepseek-v4-flash:cloud", router) == "deepseek-v4-flash"


def test_resolve_force_model_returns_none_when_router_lacks_registry():
    """Stub routers in tests don't have _registry — must not raise."""
    class _NoRegistry:
        async def ask(self, *args, **kwargs):
            return "n/a"
    assert _resolve_force_model("glm-5.1", _NoRegistry()) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_chat_completions_forces_model_when_known():
    """End-to-end: aider says `openai/glm-5.1` → router.ask receives
    force_model='glm-5.1' (bypassing classifier+ensemble)."""
    captured: dict = {}

    class _CaptureRouter:
        def __init__(self):
            self._registry = _FakeRegistry(["glm-5.1", "deepseek-v4-pro"])

        async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
            captured["force_model"] = force_model
            return "ok"

    app = build_app(_CaptureRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        await client.post("/v1/chat/completions", json={
            "model": "openai/glm-5.1",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert captured["force_model"] == "glm-5.1"


@pytest.mark.asyncio
async def test_chat_completions_does_not_force_unknown_model():
    captured: dict = {}

    class _CaptureRouter:
        def __init__(self):
            self._registry = _FakeRegistry(["glm-5.1"])

        async def ask(self, prompt, *, force_parallel=False, force_model=None, system=None):
            captured["force_model"] = force_model
            return "ok"

    app = build_app(_CaptureRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        # Some upstream "openai/gpt-4o" — fleet has no idea what that is.
        await client.post("/v1/chat/completions", json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert captured["force_model"] is None


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


# ---------- streaming heartbeat / deadline / concurrency ----------


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE byte stream into (event_name, data_dict) pairs."""
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: ") and current_event is not None:
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                payload = {}
            events.append((current_event, payload))
            current_event = None
    return events


@pytest.mark.asyncio
async def test_streaming_emits_message_start_before_router_finishes():
    """Regression guard for the v1 'fake streaming' bug — the v1 proxy
    awaited router.ask() FULLY before opening SSE, so Claude Code saw
    silence for the entire synthesis window. v2 must emit message_start
    immediately and ping while waiting."""
    # 1.5s delay: long enough that we'll see at least one ping at 5s
    # heartbeat... too long for fast CI. Use shorter heartbeat for the
    # test so we don't have to wait actual seconds.
    import fleet.proxy as proxy_mod
    original_heartbeat = proxy_mod._HEARTBEAT_INTERVAL_S
    proxy_mod._HEARTBEAT_INTERVAL_S = 0.1
    try:
        router = _StubRouter("done", delay=0.35)  # ~3 ping cycles
        app = build_app(router)  # type: ignore[arg-type]
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            assert resp.status == 200
            raw = (await resp.read()).decode("utf-8")
    finally:
        proxy_mod._HEARTBEAT_INTERVAL_S = original_heartbeat

    events = _parse_sse_events(raw)
    event_names = [name for name, _ in events]
    assert event_names[0] == "message_start"
    # At least one ping must have fired between message_start and the
    # actual content — otherwise streaming is "fake" again.
    assert "ping" in event_names, f"no heartbeat events; got {event_names}"
    ping_idx = event_names.index("ping")
    content_idx = event_names.index("content_block_start")
    assert ping_idx < content_idx, "ping should fire BEFORE content arrives"
    assert event_names[-1] == "message_stop"


@pytest.mark.asyncio
async def test_streaming_router_failure_yields_clean_stream_close():
    """If router.ask raises after headers are out, we can't switch to a
    500 — must surface the error inside the SSE body and close cleanly."""
    class _BoomRouter:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("ollama crashed mid-prompt")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status == 200
        raw = (await resp.read()).decode("utf-8")

    events = _parse_sse_events(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    # The error text should be in a content_block_delta — but GENERIC: the
    # internal exception message must NOT leak (M2). A correlation id is
    # surfaced instead so operators can triage against the server log.
    deltas = [
        d["delta"]["text"] for n, d in events
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
    ]
    joined = "".join(deltas)
    assert "router error" in joined
    assert "ollama crashed mid-prompt" not in joined
    assert "RuntimeError" not in joined
    assert re.search(r"id=[0-9a-f]{8}", joined)


@pytest.mark.asyncio
async def test_prompt_deadline_non_streaming_returns_504():
    """A router.ask that exceeds prompt_deadline_s on the non-streaming
    path must surface as 504, not as a 60s+ silent hang."""
    router = _StubRouter("never seen", delay=10.0)
    app = build_app(router, prompt_deadline_s=0.2)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 504


@pytest.mark.asyncio
async def test_prompt_deadline_streaming_closes_with_error_block():
    """Streaming path: deadline exceeded should close cleanly with an
    error message-block, not hang the connection."""
    import fleet.proxy as proxy_mod
    original_heartbeat = proxy_mod._HEARTBEAT_INTERVAL_S
    proxy_mod._HEARTBEAT_INTERVAL_S = 0.05
    try:
        router = _StubRouter("never seen", delay=10.0)
        app = build_app(router, prompt_deadline_s=0.2)  # type: ignore[arg-type]
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            assert resp.status == 200
            raw = (await resp.read()).decode("utf-8")
    finally:
        proxy_mod._HEARTBEAT_INTERVAL_S = original_heartbeat
    events = _parse_sse_events(raw)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    # Error surfaces in the stream as text content.
    deltas = [
        d["delta"]["text"] for n, d in events
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
    ]
    joined = "".join(deltas)
    assert "deadline" in joined.lower() or "timeout" in joined.lower()


@pytest.mark.asyncio
async def test_concurrent_proxy_requests_all_complete():
    """N parallel /v1/messages calls must all return their own answers
    without any cross-talk or session reuse bugs."""
    router = _StubRouter("answer", delay=0.05)
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        async def one_request(i):
            resp = await client.post("/v1/messages", json={
                "model": "x",
                "messages": [{"role": "user", "content": f"prompt {i}"}],
            })
            assert resp.status == 200
            body = await resp.json()
            return body["content"][0]["text"]

        results = await asyncio.gather(*(one_request(i) for i in range(20)))
    assert all(r == "answer" for r in results)
    assert router.call_count == 20


# ---------- M1: Host-header allowlist (DNS-rebinding / CSRF defense) ----------


class _FakeReq:
    def __init__(self, host):
        self.headers = {} if host is None else {"Host": host}


async def _ok_handler(_request):
    return "ok"


def test_host_only_extracts_hostname():
    assert _host_only("localhost:8765") == "localhost"
    assert _host_only("127.0.0.1:8765") == "127.0.0.1"
    assert _host_only("evil.com:8765") == "evil.com"
    assert _host_only("[::1]:8765") == "::1"
    assert _host_only("::1") == "::1"
    assert _host_only("LOCALHOST:8765") == "localhost"


def test_default_allowed_hosts_includes_loopback_and_bind():
    allowed = _default_allowed_hosts("192.168.1.5", 8765)
    assert "127.0.0.1:8765" in allowed
    assert "localhost:8765" in allowed
    assert "[::1]:8765" in allowed
    assert "192.168.1.5:8765" in allowed


def test_default_allowed_hosts_skips_wildcard_bind():
    allowed = _default_allowed_hosts("0.0.0.0", 8765)
    assert "0.0.0.0:8765" not in allowed
    assert "127.0.0.1:8765" in allowed


@pytest.mark.asyncio
async def test_host_guard_rejects_missing_host():
    guard = _make_host_guard({"127.0.0.1:8765"})
    with pytest.raises(web.HTTPMisdirectedRequest):
        await guard(_FakeReq(None), _ok_handler)


@pytest.mark.asyncio
async def test_host_guard_rejects_foreign_host():
    guard = _make_host_guard({"127.0.0.1:8765"})
    with pytest.raises(web.HTTPMisdirectedRequest):
        await guard(_FakeReq("evil.com:8765"), _ok_handler)


@pytest.mark.asyncio
async def test_host_guard_allows_loopback_any_port():
    guard = _make_host_guard({"127.0.0.1:8765"})
    # Ephemeral-port loopback (what the aiohttp test server uses) is allowed.
    assert await guard(_FakeReq("127.0.0.1:54321"), _ok_handler) == "ok"
    assert await guard(_FakeReq("localhost:9999"), _ok_handler) == "ok"


@pytest.mark.asyncio
async def test_host_guard_allows_explicit_allowlisted_host():
    guard = _make_host_guard({"fleet.internal:8765"})
    assert await guard(_FakeReq("fleet.internal:8765"), _ok_handler) == "ok"


@pytest.mark.asyncio
async def test_messages_rejects_foreign_host_header_421():
    router = _StubRouter("hi")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Host": "evil.com:8765"},
        )
        assert resp.status == 421


@pytest.mark.asyncio
async def test_healthz_reachable_under_host_guard():
    """The boot poller hits 127.0.0.1/healthz — must stay reachable."""
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200


@pytest.mark.asyncio
async def test_custom_allowed_hosts_honored_over_http():
    router = _StubRouter("hi")
    app = build_app(router, allowed_hosts={"fleet.internal:8765"})  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        ok = await client.post(
            "/v1/messages",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Host": "fleet.internal:8765"},
        )
        assert ok.status == 200
        bad = await client.post(
            "/v1/messages",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Host": "other.internal:8765"},
        )
        assert bad.status == 421


# ---------- M2: generic client errors, full detail server-side only ----------


def test_client_safe_error_redacts_and_correlates(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="fleet.proxy"):
        try:
            raise RuntimeError("backend at http://10.0.0.1:11434 model=secret blew up")
        except RuntimeError as exc:
            msg = _client_safe_error(exc, "unit-test")

    # Client-facing string carries NO internal detail, just a correlation id.
    assert "RuntimeError" not in msg
    assert "10.0.0.1" not in msg
    assert "secret" not in msg
    m = re.search(r"id=([0-9a-f]{8})", msg)
    assert m is not None
    corr_id = m.group(1)
    # The same id is in the server log, alongside the full traceback.
    joined = "\n".join(r.message for r in caplog.records)
    assert corr_id in joined
    assert any(r.exc_info for r in caplog.records)


@pytest.mark.asyncio
async def test_messages_non_stream_error_is_generic():
    class _BoomRouter:
        async def ask(self, *args, **kwargs):
            raise RuntimeError("ollama at /private/path crashed")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 500
        reason = resp.reason or ""
    assert "RuntimeError" not in reason
    assert "private" not in reason
    assert re.search(r"id=[0-9a-f]{8}", reason)


@pytest.mark.asyncio
async def test_chat_completions_non_stream_error_is_generic():
    class _BoomRouter:
        def __init__(self):
            self._registry = _FakeRegistry([])

        async def ask(self, *args, **kwargs):
            raise RuntimeError("model deepseek-v4-pro at backend blew up")

    app = build_app(_BoomRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 500
        reason = resp.reason or ""
    assert "RuntimeError" not in reason
    assert "deepseek" not in reason
    assert re.search(r"id=[0-9a-f]{8}", reason)


# ---------- M3: request body shape validation ----------


def test_coerce_int_falls_back_on_garbage():
    assert _coerce_int("abc", 4096) == 4096
    assert _coerce_int(None, 4096) == 4096
    assert _coerce_int("100", 4096) == 100
    assert _coerce_int(256, 4096) == 256


@pytest.mark.asyncio
async def test_messages_array_body_returns_400():
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", data=json.dumps([1, 2, 3]),
                                 headers={"Content-Type": "application/json"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_messages_string_body_returns_400():
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", data=json.dumps("just a string"),
                                 headers={"Content-Type": "application/json"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_chat_completions_array_body_returns_400():
    app = build_app(_StubRouter())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", data=json.dumps([1, 2]),
                                 headers={"Content-Type": "application/json"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_messages_non_int_max_tokens_uses_default():
    router = _StubRouter("ok")
    app = build_app(router)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/messages", json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": "abc",
        })
        assert resp.status == 200


def test_parse_request_non_int_max_tokens_falls_back():
    parsed = _parse_request({
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": "abc",
    })
    assert parsed.max_tokens == 4096


def test_parse_openai_non_int_max_tokens_falls_back():
    parsed = _parse_openai_chat_request({
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": "abc",
    })
    assert parsed.max_tokens == 4096
