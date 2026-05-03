"""Ollama provider — talks to a local Ollama HTTP server."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from fleet.providers.base import GenerateRequest, ModelInfo

logger = logging.getLogger(__name__)

_MAX_RESPONSE_CHARS = 4 * 1024 * 1024
# Default cap on simultaneous in-flight requests TO Ollama from this provider
# instance. With max-quality defaults a single prompt can fan out to ~21
# requests; under proxy load (10 concurrent /v1/messages) that's ~210
# simultaneous connections, which Ollama (and the kernel's connection table)
# will not handle gracefully. 16 is conservative; bump if your Ollama box is
# beefy and you've seen you have headroom.
_DEFAULT_MAX_CONCURRENT = 16


class OllamaProvider:
    """Provider backed by Ollama's /api/generate. One shared aiohttp session
    + a concurrency semaphore — both critical under proxy load."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        api_key: str = "",
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        # Lazily constructed inside an async context so it binds to the
        # right event loop. aiohttp sessions are loop-bound; constructing
        # here in __init__ would crash if called from sync context (CLI).
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        # Sentinel — actual Semaphore created lazily in the same async ctx
        # as the session so they share one event loop.
        self._max_concurrent = max(1, int(max_concurrent))
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _headers(self) -> dict[str, str]:
        """Return request headers including Authorization when api_key is set.

        Always include Accept: application/json — Ollama's cloud model path
        requires it or returns {"error": "unauthorized"}."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy session init under a lock — first concurrent caller wins."""
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self._timeout)
                self._session = aiohttp.ClientSession(
                    timeout=timeout, headers=self._headers(),
                )
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._session

    async def generate(self, request: GenerateRequest) -> list[Optional[str]]:
        if request.samples < 1:
            return []
        session = await self._get_session()
        # Semaphore acquired per sample (not per generate call) — the bound
        # is on simultaneous HTTP requests to Ollama, regardless of whether
        # they belong to the same prompt or different ones. This is what
        # keeps 10 concurrent proxy clients from issuing 210 simultaneous
        # connections.
        sem = self._semaphore
        assert sem is not None  # _get_session always sets it

        async def _bounded_one():
            async with sem:
                return await self._one_sample(session, request)

        tasks = [_bounded_one() for _ in range(request.samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Optional[str]] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning(
                    "ollama %s sample failed: %s", request.model, type(r).__name__
                )
                out.append(None)
            else:
                out.append(r)
        return out

    async def _one_sample(
        self,
        session: aiohttp.ClientSession,
        request: GenerateRequest,
    ) -> Optional[str]:
        payload: dict = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.system:
            payload["system"] = request.system
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        try:
            async with session.post(
                f"{self._base_url}/api/generate",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict) or "response" not in data:
                    logger.warning("missing 'response' from ollama %s", request.model)
                    return None
                response = data["response"]
                if not isinstance(response, str):
                    return None
                if len(response) > _MAX_RESPONSE_CHARS:
                    logger.warning(
                        "ollama %s response truncated to %d chars",
                        request.model, _MAX_RESPONSE_CHARS,
                    )
                    return response[:_MAX_RESPONSE_CHARS]
                return response
        except aiohttp.ClientResponseError as exc:
            logger.warning("HTTP %s from ollama %s", getattr(exc, "status", "?"), request.model)
            return None
        except aiohttp.ClientError as exc:
            logger.warning("client error from ollama %s: %s", request.model, type(exc).__name__)
            return None
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("timeout from ollama %s", request.model)
            return None

    async def list_models(self) -> list[ModelInfo]:
        # list_models is rare (startup + occasional refresh) — it gets its
        # own short-timeout session so a hung shared session can't block it.
        timeout = aiohttp.ClientTimeout(total=5)
        headers = self._headers()
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(f"{self._base_url}/api/tags") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError) as exc:
            logger.warning("ollama list_models failed: %s", exc)
            return []
        models: list[ModelInfo] = []
        for entry in data.get("models", []) or []:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("name")
            if not isinstance(raw, str) or not raw:
                continue
            models.append(ModelInfo(name=raw.split(":", 1)[0], provider=self.name))
        return models

    async def aclose(self) -> None:
        """Close the shared aiohttp session. Safe to call multiple times.
        For the proxy: registered as a cleanup so a SIGTERM doesn't leak
        keepalive connections to Ollama."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
