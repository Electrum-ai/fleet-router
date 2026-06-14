"""Discover Ollama models and map task tags to best model names."""
from __future__ import annotations

import logging

import requests

from fleet.config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Knows which model handles which task."""

    def __init__(self, config: Config):
        self._config = config
        self._available: set[str] = set()
        self._refreshed = False

    def refresh(self) -> None:
        """Fetch the set of currently-installed Ollama models.

        On any failure we record the empty set rather than pretending all
        configured models exist — dispatching to a model that isn't actually
        installed produces a guaranteed 404 downstream.
        """
        headers: dict[str, str] = {}
        if self._config.ollama.api_key:
            headers["Authorization"] = f"Bearer {self._config.ollama.api_key}"
        try:
            resp = requests.get(
                f"{self._config.ollama.base_url}/api/tags",
                timeout=5,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning(
                "ollama /api/tags unreachable (%s); registry empty", exc
            )
            self._available = set()
            self._refreshed = True
            return
        except ValueError as exc:
            logger.warning(
                "ollama /api/tags returned invalid JSON (%s); registry empty", exc
            )
            self._available = set()
            self._refreshed = True
            return

        names: set[str] = set()
        for entry in data.get("models", []) or []:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("name")
            if not isinstance(raw, str) or not raw:
                continue
            names.add(raw.split(":", 1)[0])
        self._available = names
        self._refreshed = True

    def set_available(self, names: set[str]) -> None:
        """Seed the installed-model set without a live Ollama fetch.

        Sets the refreshed flag explicitly so that an *empty* set (a valid
        state — local Ollama has nothing pulled) is honored rather than
        falling through to a network refresh. Tests use this instead of
        poking `_available` directly."""
        self._available = set(names)
        self._refreshed = True

    def _ensure_refreshed(self) -> None:
        # Honor a pre-populated registry (e.g. via set_available, or a direct
        # `_available` poke in legacy tests) and skip the live fetch. The
        # `_refreshed` flag is the authoritative signal; the truthiness check
        # on `_available` keeps older direct-poke callers working.
        if self._refreshed or self._available:
            return
        self.refresh()

    def get_best_for_tag(self, tag: str) -> str | None:
        """Return single best model for a tag, or None if no match."""
        self._ensure_refreshed()
        candidates = self.models_for_tag(tag, top_n=1)
        return candidates[0] if candidates else None

    def _is_cloud_model(self, name: str, entry) -> bool:
        """Cloud models (e.g. deepseek-v4-pro:cloud) are hosted remotely and
        never appear in the local Ollama /api/tags list — they should not be
        filtered by local availability."""
        return ":cloud" in (entry.api_model or name)

    def models_for_tag(self, tag: str, top_n: int | None = None) -> list[str]:
        """Return up to `top_n` available models matching a tag, sorted by
        ascending priority. Defaults to `config.thresholds.max_parallel`.

        Ollama-provider models are gated on the live `_available` set so we
        never dispatch to models that aren't actually installed. Hosted
        providers (openai, anthropic, etc.) and cloud-tagged models are trusted
        from config — there is no per-account model list to gate against."""
        self._ensure_refreshed()
        if top_n is None:
            top_n = self._config.thresholds.max_parallel
        scored: list[tuple[str, int]] = []
        for name, entry in self._config.models.items():
            if tag not in entry.tags:
                continue
            if entry.provider == "ollama" and name not in self._available and not self._is_cloud_model(name, entry):
                continue
            scored.append((name, entry.priority))
        scored.sort(key=lambda x: x[1])
        return [name for name, _ in scored[:top_n]]

    def all_models_for_tag(self, tag: str) -> list[str]:
        """Every available model matching a tag, priority-sorted, no top_n cap.
        Used by bandit-aware selection so the bandit can explore beyond
        `max_parallel` head-of-line candidates."""
        return self.models_for_tag(tag, top_n=10_000)

    def all_available(self) -> list[str]:
        """List currently available models, sorted by configured priority
        (unknown models sort last alphabetically). Includes both Ollama
        models that are actually installed AND non-Ollama configured models
        AND cloud-tagged Ollama models (hosted remotely)."""
        self._ensure_refreshed()
        configured = self._config.models
        ollama_installed = set(self._available)
        names: set[str] = set(ollama_installed)
        for name, entry in configured.items():
            if entry.provider != "ollama":
                names.add(name)
            elif self._is_cloud_model(name, entry):
                names.add(name)
        return sorted(
            names,
            key=lambda n: (
                configured[n].priority if n in configured else 99,
                n,
            ),
        )
