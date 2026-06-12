"""Provider pool — currently holds only the Ollama provider.

Kept as a thin abstraction so testability is uniform (Verifier/Classifier
accept any Provider) and future Ollama-compatible backends (vLLM, LM Studio
with the Ollama API surface) can be registered without touching call sites.
"""
from __future__ import annotations

import logging
from typing import Optional

from fleet.config import Config
from fleet.providers.base import Provider
from fleet.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


class ProviderPool:
    """Holds initialized Provider instances keyed by name."""

    def __init__(self, providers: Optional[dict[str, Provider]] = None):
        self._providers: dict[str, Provider] = dict(providers or {})

    def register(self, provider: Provider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> Optional[Provider]:
        return self._providers.get(name)

    def names(self) -> list[str]:
        return sorted(self._providers.keys())

    @classmethod
    def from_config(cls, config: Config) -> "ProviderPool":
        # Size the shared session to the CHAT budget. aiohttp's per-request
        # ClientTimeout *fully replaces* the session timeout (it is not a cap):
        # `real_timeout = timeout if timeout is not sentinel else self._timeout`.
        # So the dispatcher's per-request reasoning override (240s) takes full
        # effect regardless of this value — and calls that set NO per-request
        # timeout (the shared judge/classifier on this provider) correctly
        # inherit the 60s chat default instead of a bloated max budget.
        timeouts = config.thresholds.timeouts
        timeout = (
            timeouts.get("chat", config.thresholds.parallel_timeout)
            if timeouts
            else config.thresholds.parallel_timeout
        )
        pool = cls()
        pool.register(OllamaProvider(
            base_url=config.ollama.base_url,
            timeout=timeout,
            api_key=config.ollama.api_key,
        ))
        return pool

    async def aclose_all(self) -> None:
        for p in self._providers.values():
            try:
                await p.aclose()
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning("provider %s aclose failed: %s", p.name, exc)
