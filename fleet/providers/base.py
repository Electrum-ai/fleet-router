"""Provider Protocol — uniform interface across Ollama / OpenAI / Anthropic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerateRequest:
    """Single request to a provider. `samples > 1` enables self-consistency:
    the provider returns N independent samples (with temperature > 0) which
    can be aggregated by the synthesizer for majority voting."""
    model: str
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    samples: int = 1
    # Per-request wall-clock budget (seconds). When set, overrides the
    # provider's session-wide default so a slow reasoning model gets its
    # larger class budget instead of being cut off at the chat budget.
    # None = fall back to the session default.
    timeout: Optional[float] = None


@dataclass(frozen=True)
class ModelInfo:
    name: str          # canonical short name
    provider: str      # "ollama" | "openai" | "anthropic" | ...


@runtime_checkable
class Provider(Protocol):
    """Async LLM provider. Returns one entry per requested sample; failed
    samples are returned as None (do not raise)."""

    name: str

    async def generate(self, request: GenerateRequest) -> list[Optional[str]]:
        ...

    async def list_models(self) -> list[ModelInfo]:
        ...

    async def aclose(self) -> None:
        ...
