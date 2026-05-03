"""Anthropic Messages API-compatible HTTP proxy backed by FleetRouter.

Lets `claude` (Claude Code CLI) talk to fleet → Ollama instead of Anthropic:

    fleet serve --port 8765 &
    export ANTHROPIC_BASE_URL=http://localhost:8765
    export ANTHROPIC_API_KEY=fleet-local
    claude

Implements the subset of the Messages API that Claude Code uses for chat:
- POST /v1/messages (streaming and non-streaming)
- GET  /healthz

Tool use (`tools`/`tool_use`/`tool_result` blocks) is NOT translated —
Anthropic's tool format does not map cleanly to Ollama's OpenAI-style
function calling. Tool blocks in the input are flattened into text so the
underlying model at least sees the conversation; the model's reply is
returned as a single text block. This makes plain chat work; agentic tool
loops will not.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from aiohttp import web

from fleet.router import FleetRouter

logger = logging.getLogger(__name__)

# How many characters the SSE writer emits per content_block_delta event.
# Smaller = smoother UX, more events. 80 is a reasonable balance — Claude
# Code's renderer batches deltas anyway.
_STREAM_CHUNK_CHARS = 80


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


def _approx_tokens(text: str) -> int:
    """Cheap token estimate. Anthropic clients display these but don't
    enforce them — accuracy isn't critical."""
    return max(1, len(text) // 4)


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


async def _stream_anthropic(
    text: str, requested_model: str, message_id: str, prompt_tokens: int
) -> AsyncIterator[bytes]:
    """Emit the Anthropic streaming event sequence for a single text block.

    Sequence: message_start → content_block_start → N×content_block_delta
    → content_block_stop → message_delta → message_stop.
    """
    out_tokens = _approx_tokens(text)

    yield _sse("message_start", {
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


def build_app(router: FleetRouter, *, api_key: Optional[str] = None) -> web.Application:
    """Construct the aiohttp app. `api_key`, if set, is required as the
    `x-api-key` header — basic guard against a stray local request hitting
    your fleet from elsewhere on the network."""
    app = web.Application()

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "service": "fleet-proxy"})

    async def messages(request: web.Request) -> web.StreamResponse:
        if api_key:
            presented = request.headers.get("x-api-key") or request.headers.get("authorization", "")
            presented = presented.removeprefix("Bearer ").strip()
            if presented != api_key:
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

        try:
            answer = await router.ask(parsed.prompt, system=parsed.system)
        except Exception as exc:  # noqa: BLE001 — surface any router failure as 500
            logger.exception("router.ask failed")
            raise web.HTTPInternalServerError(
                reason=f"router error: {type(exc).__name__}: {exc}"
            )

        # Router can return dict[model -> text] when caller forces parallel,
        # but our /v1/messages path never forces it. Defensive flatten.
        if isinstance(answer, dict):
            answer = "\n\n".join(f"--- {m} ---\n{t}" for m, t in answer.items())

        if not parsed.stream:
            return web.json_response(_build_message_response(
                answer, parsed.requested_model, message_id, prompt_tokens,
            ))

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        async for chunk in _stream_anthropic(
            answer, parsed.requested_model, message_id, prompt_tokens
        ):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app.router.add_get("/healthz", healthz)
    app.router.add_post("/v1/messages", messages)
    return app


def serve(
    router: FleetRouter,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    api_key: Optional[str] = None,
) -> None:
    """Blocking serve — used by the CLI."""
    app = build_app(router, api_key=api_key)
    logger.info("fleet-proxy listening on http://%s:%d", host, port)
    web.run_app(app, host=host, port=port, print=None)
