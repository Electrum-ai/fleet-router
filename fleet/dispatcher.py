"""Async parallel dispatch to Ollama /api/generate."""
from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp

from fleet.config import Config


class EnsembleDispatcher:
    """Dispatch prompts to one or more Ollama models in parallel."""

    def __init__(self, config: Config):
        self._base_url = config.ollama.base_url.rstrip("/")
        self._timeout = config.thresholds.parallel_timeout

    async def run(
        self,
        prompt: str,
        models: list[str],
        system: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Run prompt against all models, return {model_name: response_or_None}."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call(session, model, prompt, system)
                for model in models
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, Optional[str]] = {}
        for model, result in zip(models, results):
            if isinstance(result, Exception):
                output[model] = None
            else:
                output[model] = result
        return output

    async def _call(
        self,
        session: aiohttp.ClientSession,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Optional[str]:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            async with session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("response")
        except Exception:
            return None
