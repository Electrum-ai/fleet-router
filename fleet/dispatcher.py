"""Multi-provider dispatch with self-consistency support.

`EnsembleDispatcher.run` keeps the legacy 1-sample-per-model API for backward
compatibility. `run_multi` returns N samples per model for self-consistency
and judge-based synthesis.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fleet.config import Config
from fleet.providers.base import GenerateRequest, Provider
from fleet.providers.ollama import OllamaProvider
from fleet.providers.pool import ProviderPool

logger = logging.getLogger(__name__)


class EnsembleDispatcher:
    """Routes each model to its configured provider and dispatches in parallel."""

    def __init__(
        self,
        config: Config,
        pool: Optional[ProviderPool] = None,
    ):
        self._config = config
        self._pool = pool or ProviderPool.from_config(config)
        # The shared session is sized to the CHAT budget — it is only the
        # DEFAULT for requests that don't set their own timeout. aiohttp's
        # per-request ClientTimeout fully replaces (not caps) the session
        # value, so the reasoning class still gets its full 240s via the
        # per-request GenerateRequest.timeout set below.
        timeouts = config.thresholds.timeouts
        self._timeout = (
            timeouts.get("chat", config.thresholds.parallel_timeout)
            if timeouts
            else config.thresholds.parallel_timeout
        )
        # Fallback provider for models that aren't in config (test compat,
        # ad-hoc CLI use). Always Ollama with the configured base_url.
        self._default_provider: Provider = (
            self._pool.get("ollama")
            or OllamaProvider(
                base_url=config.ollama.base_url,
                timeout=self._timeout,
                api_key=config.ollama.api_key,
            )
        )

    async def run(
        self,
        prompt: str,
        models: list[str],
        system: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Single sample per model — backward-compat API."""
        multi = await self.run_multi(prompt, models, samples=1, system=system)
        return {m: (samples[0] if samples else None) for m, samples in multi.items()}

    async def run_multi(
        self,
        prompt: str,
        models: list[str],
        samples: int = 1,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> dict[str, list[str]]:
        """Returns each model's list of valid (non-None) samples. Empty list
        means every sample failed."""
        if not models:
            return {}
        timeouts = self._config.thresholds.timeouts
        chat_budget = timeouts.get("chat", self._timeout)
        plan: list[tuple[str, Provider, GenerateRequest]] = []
        for name in models:
            entry = self._config.models.get(name)
            if entry is not None:
                provider = self._pool.get(entry.provider) or self._default_provider
                api_model = entry.api_model or name
                model_class = entry.model_class
            else:
                # Unconfigured / ad-hoc models default to the chat budget — a
                # reasoning model must be declared `class: reasoning` to earn
                # the larger timeout.
                provider = self._default_provider
                api_model = name
                model_class = "chat"
            req = GenerateRequest(
                model=api_model,
                prompt=prompt,
                system=system,
                temperature=temperature,
                samples=samples,
                timeout=timeouts.get(model_class, chat_budget),
            )
            plan.append((name, provider, req))

        results = await asyncio.gather(
            *(provider.generate(req) for _, provider, req in plan),
            return_exceptions=True,
        )

        out: dict[str, list[str]] = {}
        for (name, _, _), result in zip(plan, results):
            if isinstance(result, BaseException):
                logger.warning("model %s dispatch crashed: %s", name, type(result).__name__)
                out[name] = []
            else:
                out[name] = [s for s in result if isinstance(s, str) and s]
        return out

    async def aclose(self) -> None:
        await self._pool.aclose_all()
