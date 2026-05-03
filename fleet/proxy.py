"""HTTP proxy backed by FleetRouter, speaking two API dialects.

Anthropic Messages API — for `claude` (Claude Code CLI):

    fleet serve --port 8765 &
    export ANTHROPIC_BASE_URL=http://localhost:8765
    export ANTHROPIC_API_KEY=fleet-local
    claude

OpenAI Chat Completions API — for aider, OpenAI SDKs, llama.cpp UIs,
and anything else that speaks /v1/chat/completions:

    export OPENAI_API_BASE=http://localhost:8765/v1
    export OPENAI_API_KEY=fleet-local
    aider --model fleet-router

Endpoints:
- POST /v1/messages              (Anthropic, streaming + non-streaming)
- POST /v1/chat/completions      (OpenAI, streaming + non-streaming)
- GET  /v1/models                (OpenAI-style listing)
- GET  /healthz

Tool / function calling is NOT translated in either dialect — neither
Anthropic's `tool_use`/`tool_result` blocks nor OpenAI's `tool_calls`
function-calling JSON map cleanly onto fleet's single-prompt-in,
single-answer-out interface. Tool blocks in the input are flattened
into text so the underlying model at least sees the conversation;
the reply is returned as a single text block. Plain chat works;
agentic tool loops do not.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from aiohttp import web

from fleet.router import (
    ERROR_ALL_MODELS_FAILED,
    ERROR_MODEL_FAILED,
    ERROR_NO_MODEL,
    ERROR_NO_MODELS,
    FleetRouter,
)

logger = logging.getLogger(__name__)

# Sentinel responses from router.ask that indicate Ollama is unreachable or
# misconfigured. router.ask never raises for these — it returns the string
# directly — so the proxy has to pattern-match to attach an actionable hint.
_OLLAMA_DOWN_SENTINELS: tuple[str, ...] = (
    ERROR_ALL_MODELS_FAILED,
    ERROR_MODEL_FAILED,
    ERROR_NO_MODEL,
    ERROR_NO_MODELS,
)

_OLLAMA_DOWN_HINT = (
    "\n\nFleet Router could not reach Ollama. Check:\n"
    "  • `ollama serve` is running\n"
    "  • `curl http://localhost:11434/api/tags` responds\n"
    "  • models in fleet/config.yaml are pulled (`ollama pull <name>`)\n"
)


def _maybe_enrich_with_ollama_hint(text: str) -> str:
    """If `text` is a router error sentinel, append the troubleshooting hint.
    Cheap prefix check — sentinels all start with '(' and are short."""
    if not text.startswith("("):
        return text
    for sentinel in _OLLAMA_DOWN_SENTINELS:
        if text.startswith(sentinel):
            return text + _OLLAMA_DOWN_HINT
    return text

# How many characters the SSE writer emits per content_block_delta event.
# Smaller = smoother UX, more events. 80 is a reasonable balance — Claude
# Code's renderer batches deltas anyway.
_STREAM_CHUNK_CHARS = 80

# Heartbeat cadence while waiting for router.ask to complete. With max-quality
# defaults a single prompt can run 30-90s (3 models × 7 samples + judge +
# escalation + refinement); without heartbeats, Claude Code's HTTP client and
# any intermediate proxy will time out the connection. Anthropic's spec allows
# `ping` events at any time during a stream, which keeps the TCP connection
# warm and signals to the SDK that the server is still working.
_HEARTBEAT_INTERVAL_S = 5.0

# Hard cap on a single prompt's wall-clock from the proxy's perspective. A
# stuck Ollama call could otherwise hold a dispatcher slot indefinitely.
# 10 minutes is generous for max-quality (refinement + escalation can be slow);
# operators with tighter SLOs should override.
_DEFAULT_PROMPT_DEADLINE_S = 600.0


@dataclass
class _ParsedRequest:
    """Internal representation of an Anthropic /v1/messages request after
    we've flattened it into something fleet can consume."""

    prompt: str
    system: Optional[str]
    stream: bool
    requested_model: str  # echo back in response.model
    max_tokens: int


def _flatten_content(content: Any) -> str:
    """Anthropic message content can be a string OR a list of typed blocks.
    Flatten to plain text — keeping tool blocks as readable summaries so the
    model retains context even though it can't act on them."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            name = block.get("name", "?")
            inp = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"[tool_call name={name} input={inp}]")
        elif btype == "tool_result":
            tid = block.get("tool_use_id", "?")
            inner = block.get("content", "")
            inner_text = _flatten_content(inner) if not isinstance(inner, str) else inner
            parts.append(f"[tool_result id={tid}]\n{inner_text}")
        elif btype == "image":
            parts.append("[image omitted — fleet/Ollama text-only path]")
        else:
            # Unknown block type — preserve raw so the model sees something.
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def _parse_request(body: dict) -> _ParsedRequest:
    """Translate Anthropic Messages API JSON → ParsedRequest.

    Concatenates the message history into a single prompt with role markers.
    This is lossy vs. true multi-turn chat but keeps fleet's interface
    (single prompt → single answer) unchanged."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise web.HTTPBadRequest(reason="messages: required non-empty array")

    system = body.get("system")
    if isinstance(system, list):
        # System can also be a list of content blocks.
        system = _flatten_content(system)
    elif system is not None and not isinstance(system, str):
        system = str(system)

    turns: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        text = _flatten_content(msg.get("content", ""))
        if not text.strip():
            continue
        # Role markers help the model distinguish turns when we collapse
        # the conversation into a single prompt.
        marker = "Human" if role == "user" else "Assistant"
        turns.append(f"{marker}: {text}")
    # Cue the model to produce the next assistant turn.
    turns.append("Assistant:")
    prompt = "\n\n".join(turns)

    return _ParsedRequest(
        prompt=prompt,
        system=system,
        stream=bool(body.get("stream", False)),
        requested_model=str(body.get("model", "fleet-router")),
        max_tokens=int(body.get("max_tokens", 4096)),
    )


def _parse_openai_chat_request(body: dict) -> _ParsedRequest:
    """Translate OpenAI Chat Completions JSON → ParsedRequest.

    System messages (role=system) are extracted and concatenated into
    `parsed.system`; user/assistant turns collapse into the prompt the
    same way as Anthropic. Tool/function-call payloads are flattened to
    readable text — see module docstring for the limitation."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise web.HTTPBadRequest(reason="messages: required non-empty array")

    system_parts: list[str] = []
    turns: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        # OpenAI content can be string OR list of {type:"text"|"image_url",...}.
        # Reuse _flatten_content — text blocks render naturally; image and
        # tool_call/tool_result are summarized so the model sees something.
        text = _flatten_content(msg.get("content", ""))
        # OpenAI tool_calls live OUTSIDE content as a sibling field on the
        # assistant message — surface them so the model retains context.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                text = (text + "\n" if text else "") + f"[tool_call name={name} args={args}]"
        if role == "tool":
            # tool/function results have a `tool_call_id` sibling field.
            tid = msg.get("tool_call_id", "?")
            text = f"[tool_result id={tid}]\n{text}"
        if not text.strip():
            continue
        if role == "system":
            system_parts.append(text)
            continue
        marker = "Human" if role in ("user", "tool") else "Assistant"
        turns.append(f"{marker}: {text}")
    turns.append("Assistant:")

    return _ParsedRequest(
        prompt="\n\n".join(turns),
        system="\n\n".join(system_parts) if system_parts else None,
        stream=bool(body.get("stream", False)),
        requested_model=str(body.get("model", "fleet-router")),
        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
    )


def _approx_tokens(text: str) -> int:
    """Cheap token estimate. Anthropic clients display these but don't
    enforce them — accuracy isn't critical."""
    return max(1, len(text) // 4)


def _coerce_answer_to_string(answer: Any) -> str:
    """Both endpoints need to handle dict returns from router.ask() (parallel
    mode without synthesis). Render as labeled sections to preserve info."""
    if isinstance(answer, dict):
        return "\n\n".join(f"--- {m} ---\n{t}" for m, t in answer.items())
    return answer if isinstance(answer, str) else str(answer)


def _build_chat_completion_response(
    text: str, requested_model: str, completion_id: str, prompt_tokens: int,
) -> dict:
    """OpenAI Chat Completions non-streaming response shape."""
    out_tokens = _approx_tokens(text)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": prompt_tokens + out_tokens,
        },
    }


def _openai_chunk(completion_id: str, model: str, delta: dict, finish: Optional[str] = None) -> bytes:
    """One OpenAI streaming chunk frame (SSE `data:` line)."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _openai_done() -> bytes:
    """OpenAI's terminal sentinel — clients close the stream on this."""
    return b"data: [DONE]\n\n"


def _openai_heartbeat() -> bytes:
    """SSE comment line — kept-alive marker that conforming clients ignore.
    OpenAI's spec doesn't define a heartbeat event, but `:` comment lines
    are part of the SSE protocol and are skipped by all standard parsers."""
    return b": keep-alive\n\n"


async def _stream_openai_body(
    text: str, requested_model: str, completion_id: str,
) -> AsyncIterator[bytes]:
    """Emit `role: assistant` chunk → N×content chunks → final `finish_reason`
    chunk → [DONE]. Caller is expected to flush the answer ONLY after
    router.ask resolves; heartbeats during the wait happen separately."""
    yield _openai_chunk(completion_id, requested_model, {"role": "assistant"})
    for i in range(0, len(text), _STREAM_CHUNK_CHARS):
        chunk = text[i:i + _STREAM_CHUNK_CHARS]
        yield _openai_chunk(completion_id, requested_model, {"content": chunk})
    yield _openai_chunk(completion_id, requested_model, {}, finish="stop")
    yield _openai_done()


def _build_message_response(
    text: str, requested_model: str, message_id: str, prompt_tokens: int
) -> dict:
    """Anthropic Messages API non-streaming response shape."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": _approx_tokens(text),
        },
    }


def _sse(event: str, data: dict) -> bytes:
    """Format one Server-Sent Event in the way Anthropic's SDK expects."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _message_start_event(
    requested_model: str, message_id: str, prompt_tokens: int
) -> bytes:
    """Pre-compute the message_start frame so the proxy can flush it
    immediately on connection — before router.ask runs — to keep the SDK
    from timing out the connection during the long synthesis phase."""
    return _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    })


def _ping_event() -> bytes:
    return _sse("ping", {"type": "ping"})


async def _stream_anthropic_body(text: str) -> AsyncIterator[bytes]:
    """Emit the post-message_start sequence: content_block_start →
    N×content_block_delta → content_block_stop → message_delta → message_stop.

    Caller is responsible for emitting the message_start frame BEFORE
    awaiting whatever produces `text` (so the connection stays warm)."""
    out_tokens = _approx_tokens(text)
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    for i in range(0, len(text), _STREAM_CHUNK_CHARS):
        chunk = text[i:i + _STREAM_CHUNK_CHARS]
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def build_app(
    router: FleetRouter,
    *,
    api_key: Optional[str] = None,
    prompt_deadline_s: float = _DEFAULT_PROMPT_DEADLINE_S,
) -> web.Application:
    """Construct the aiohttp app. `api_key`, if set, is required as the
    `x-api-key` header — basic guard against a stray local request hitting
    your fleet from elsewhere on the network. `prompt_deadline_s` caps
    how long a single /v1/messages request will wait for router.ask
    before giving up and returning a structured error."""
    app = web.Application()

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "service": "fleet-proxy"})

    async def list_models(_request: web.Request) -> web.Response:
        """OpenAI-style model listing — handy for `curl` debugging and any
        OpenAI-compatible client probing the proxy. Claude Code itself takes
        the model from the request body, so this isn't on its hot path."""
        try:
            names = list(router._registry.all_available())  # type: ignore[attr-defined]
        except AttributeError:
            names = []
        now = int(time.time())
        return web.json_response({
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": now, "owned_by": "fleet"}
                for name in names
            ],
        })

    async def messages(request: web.Request) -> web.StreamResponse:
        if api_key:
            presented = request.headers.get("x-api-key") or request.headers.get("authorization", "")
            presented = presented.removeprefix("Bearer ").strip()
            # Constant-time compare so a network attacker on --host 0.0.0.0
            # can't recover the key byte-by-byte from response timing.
            if not hmac.compare_digest(presented, api_key):
                raise web.HTTPUnauthorized(reason="invalid x-api-key")

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}")

        parsed = _parse_request(body)
        prompt_tokens = _approx_tokens(parsed.prompt)
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        logger.info(
            "proxy: model=%s stream=%s prompt_chars=%d",
            parsed.requested_model, parsed.stream, len(parsed.prompt),
        )

        # Non-streaming path: nothing to keep alive, just await + JSON respond.
        if not parsed.stream:
            try:
                answer = await asyncio.wait_for(
                    router.ask(parsed.prompt, system=parsed.system),
                    timeout=prompt_deadline_s,
                )
            except asyncio.TimeoutError:
                raise web.HTTPGatewayTimeout(
                    reason=f"router.ask exceeded {prompt_deadline_s}s deadline"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("router.ask failed")
                raise web.HTTPInternalServerError(
                    reason=f"router error: {type(exc).__name__}: {exc}"
                )
            if isinstance(answer, dict):
                answer = "\n\n".join(f"--- {m} ---\n{t}" for m, t in answer.items())
            answer = _maybe_enrich_with_ollama_hint(answer)
            return web.json_response(_build_message_response(
                answer, parsed.requested_model, message_id, prompt_tokens,
            ))

        # Streaming path: open SSE BEFORE awaiting router.ask, send
        # message_start immediately, then ping every _HEARTBEAT_INTERVAL_S
        # while the synthesis pipeline runs. Without this the connection
        # appears dead for 30-90s under max-quality defaults — Claude Code
        # / proxies / load balancers will time it out.
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        await resp.write(_message_start_event(
            parsed.requested_model, message_id, prompt_tokens,
        ))

        ask_task = asyncio.create_task(
            router.ask(parsed.prompt, system=parsed.system)
        )
        deadline = asyncio.get_event_loop().time() + prompt_deadline_s
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    ask_task.cancel()
                    raise asyncio.TimeoutError(
                        f"router.ask exceeded {prompt_deadline_s}s deadline"
                    )
                tick = min(_HEARTBEAT_INTERVAL_S, remaining)
                try:
                    # Race the synthesis task against the heartbeat tick.
                    # asyncio.shield prevents the wait_for timeout from
                    # cancelling ask_task — we only want the wait to expire.
                    answer = await asyncio.wait_for(
                        asyncio.shield(ask_task), timeout=tick,
                    )
                    break
                except asyncio.TimeoutError:
                    if ask_task.done():
                        # The shield consumed our wait window AND the task
                        # finished — pull its result on the next loop iter.
                        continue
                    await resp.write(_ping_event())
        except Exception as exc:  # noqa: BLE001
            logger.exception("router.ask failed mid-stream")
            # Surface the failure as a text block + clean stream close.
            # We can't switch to HTTP 500 once headers are out — the SDK
            # would see a half-stream and treat it as a network error,
            # which is worse than a structured error message.
            err_text = f"(router error: {type(exc).__name__}: {exc})"
            async for chunk in _stream_anthropic_body(err_text):
                await resp.write(chunk)
            await resp.write_eof()
            return resp

        if isinstance(answer, dict):
            answer = "\n\n".join(f"--- {m} ---\n{t}" for m, t in answer.items())
        answer = _maybe_enrich_with_ollama_hint(answer)
        async for chunk in _stream_anthropic_body(answer):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        """OpenAI Chat Completions endpoint — for aider, openai SDK,
        llama.cpp UIs, and any other client speaking /v1/chat/completions.
        Mirrors `messages` (Anthropic) structurally; differs only in
        request parsing and SSE chunk format."""
        if api_key:
            # OpenAI clients send Authorization: Bearer <key>; some quirky
            # ones still send x-api-key. Accept both.
            presented = request.headers.get("authorization", "") or request.headers.get("x-api-key", "")
            presented = presented.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(presented, api_key):
                raise web.HTTPUnauthorized(reason="invalid api key")

        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(reason=f"invalid JSON: {exc}")

        parsed = _parse_openai_chat_request(body)
        prompt_tokens = _approx_tokens(parsed.prompt)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        logger.info(
            "proxy[openai]: model=%s stream=%s prompt_chars=%d",
            parsed.requested_model, parsed.stream, len(parsed.prompt),
        )

        if not parsed.stream:
            try:
                answer = await asyncio.wait_for(
                    router.ask(parsed.prompt, system=parsed.system),
                    timeout=prompt_deadline_s,
                )
            except asyncio.TimeoutError:
                raise web.HTTPGatewayTimeout(
                    reason=f"router.ask exceeded {prompt_deadline_s}s deadline"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("router.ask failed (openai endpoint)")
                raise web.HTTPInternalServerError(
                    reason=f"router error: {type(exc).__name__}: {exc}"
                )
            answer = _maybe_enrich_with_ollama_hint(_coerce_answer_to_string(answer))
            return web.json_response(_build_chat_completion_response(
                answer, parsed.requested_model, completion_id, prompt_tokens,
            ))

        # Streaming path — same pattern as Anthropic: open SSE, fire one
        # frame immediately, then heartbeat (SSE comment line) every
        # _HEARTBEAT_INTERVAL_S until router.ask resolves. The first
        # `data: {role: assistant}` chunk is the OpenAI equivalent of
        # message_start — clients use it to render the empty bubble.
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        await resp.write(_openai_chunk(
            completion_id, parsed.requested_model, {"role": "assistant"},
        ))

        ask_task = asyncio.create_task(
            router.ask(parsed.prompt, system=parsed.system)
        )
        deadline = asyncio.get_event_loop().time() + prompt_deadline_s
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    ask_task.cancel()
                    raise asyncio.TimeoutError(
                        f"router.ask exceeded {prompt_deadline_s}s deadline"
                    )
                tick = min(_HEARTBEAT_INTERVAL_S, remaining)
                try:
                    answer = await asyncio.wait_for(
                        asyncio.shield(ask_task), timeout=tick,
                    )
                    break
                except asyncio.TimeoutError:
                    if ask_task.done():
                        continue
                    await resp.write(_openai_heartbeat())
        except Exception as exc:  # noqa: BLE001
            logger.exception("router.ask failed mid-stream (openai)")
            err_text = f"(router error: {type(exc).__name__}: {exc})"
            # Reuse the streaming body — emit the error as content so
            # clients see structured failure text, then [DONE].
            for i in range(0, len(err_text), _STREAM_CHUNK_CHARS):
                await resp.write(_openai_chunk(
                    completion_id, parsed.requested_model,
                    {"content": err_text[i:i + _STREAM_CHUNK_CHARS]},
                ))
            await resp.write(_openai_chunk(
                completion_id, parsed.requested_model, {}, finish="stop",
            ))
            await resp.write(_openai_done())
            await resp.write_eof()
            return resp

        answer = _maybe_enrich_with_ollama_hint(_coerce_answer_to_string(answer))
        # Skip the role chunk we already sent — start at the content chunks.
        for i in range(0, len(answer), _STREAM_CHUNK_CHARS):
            await resp.write(_openai_chunk(
                completion_id, parsed.requested_model,
                {"content": answer[i:i + _STREAM_CHUNK_CHARS]},
            ))
        await resp.write(_openai_chunk(
            completion_id, parsed.requested_model, {}, finish="stop",
        ))
        await resp.write(_openai_done())
        await resp.write_eof()
        return resp

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def serve(
    router: FleetRouter,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    api_key: Optional[str] = None,
    prompt_deadline_s: float = _DEFAULT_PROMPT_DEADLINE_S,
) -> None:
    """Blocking serve — used by the CLI."""
    app = build_app(router, api_key=api_key, prompt_deadline_s=prompt_deadline_s)
    logger.info(
        "fleet-proxy listening on http://%s:%d (deadline=%.0fs)",
        host, port, prompt_deadline_s,
    )
    web.run_app(app, host=host, port=port, print=None)
