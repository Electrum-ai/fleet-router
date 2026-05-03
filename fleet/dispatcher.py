"""Async parallel dispatch to Ollama /api/generate."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from fleet.config import Config

logger = logging.getLogger(__name__)


class EnsembleDispatcher:
    """Dispatch prompts to one or more Ollama models in parallel."""

    def __init__(self, config: Config):
        self._base_url = config.ollama.base_url.rstrip("/")
        self._timeout = config.thresholds.parallel_timeout

    async def run(
        self,
        prompt: str,
        models: list[str],
        system: str | None = None,
    ) -> dict[str, str | None]:
        """Run prompt against all models, return {model_name: response_or_None}."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call(session, model, prompt, system)
                for model in models
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, str | None] = {}
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
        system: str | None = None,
    ) -> str | None:
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
                await resp.raise_for_status()
                data = await resp.json()
                if "response" not in data:
                    logger.warning("Missing 'response' key in Ollama response for model %s", model)
                    return None
                return data["response"]
        except aiohttp.ClientResponseError:
            logger.exception("Ollama HTTP error for model %s", model)
            return None
        except aiohttp.ClientError:
            logger.exception("Ollama client error for model %s", model)
            return None
        except asyncio.TimeoutError:
            logger.exception("Ollama request timed out for model %s", model)
            return None
        except ValueError:
            logger.exception("Ollama JSON decode error for model %s", model)
            return None
