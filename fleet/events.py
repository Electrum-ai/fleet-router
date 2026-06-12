"""Typed event bus for router observability.

Subscribers plug in via `bus.subscribe(callable)`. Default = no subscribers,
events are no-ops. Sinks (LoggingSink, JSONLSink, PrometheusSink) live in
this module or as user-supplied callables.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouterEvent:
    ts: float = field(default_factory=time.time)


@dataclass
class PromptClassified(RouterEvent):
    tag: str = ""
    confidence: float = 0.0
    prompt: str = ""


@dataclass
class ModelDispatched(RouterEvent):
    models: list[str] = field(default_factory=list)
    tag: str = ""
    samples: int = 1


@dataclass
class ModelCompleted(RouterEvent):
    model: str = ""
    latency_ms: int = 0
    tokens: int = 0
    error: str = ""


@dataclass
class ResponseSynthesized(RouterEvent):
    tag: str = ""
    mode: str = ""
    winner_model: Optional[str] = None
    winner_score: Optional[float] = None
    abstain: bool = False


# Subscribers receive events; they should not raise — exceptions are swallowed.
Sink = Callable[[RouterEvent], None]


class EventBus:
    """Synchronous publish/subscribe — sinks are called inline. For
    high-throughput async telemetry, sinks should hand off to a queue."""

    def __init__(self):
        self._sinks: list[Sink] = []

    def subscribe(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def unsubscribe(self, sink: Sink) -> None:
        """Remove a previously-subscribed sink. No-op if it isn't subscribed.
        Lets a short-lived consumer (e.g. the eval harness's calibration
        collector) attach for one run and detach cleanly afterwards instead of
        accumulating stale sinks on a long-lived router's bus."""
        try:
            self._sinks.remove(sink)
        except ValueError:
            pass

    def emit(self, event: RouterEvent) -> None:
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("event sink raised: %s", exc)


def logging_sink(event: RouterEvent) -> None:
    logger.info("event %s: %s", type(event).__name__, event)


def cli_progress_sink(event: RouterEvent) -> None:
    """One-line stderr progress for interactive CLI use.

    Without this, max-quality runs (3 models × N samples + verifier +
    judge + escalation/refinement) take 60–180s with zero feedback —
    the user has no signal that work is happening. These lines go to
    stderr so they don't pollute the answer on stdout."""
    import sys

    if isinstance(event, PromptClassified):
        print(
            f"→ classified as {event.tag!r} (confidence {event.confidence:.2f})",
            file=sys.stderr, flush=True,
        )
    elif isinstance(event, ModelDispatched):
        models = ", ".join(event.models)
        print(
            f"→ dispatching {len(event.models)} model(s) × {event.samples} sample(s): {models}",
            file=sys.stderr, flush=True,
        )
    elif isinstance(event, ResponseSynthesized):
        if event.abstain:
            tail = "abstained"
        elif event.winner_model is not None:
            score = f"{event.winner_score:.2f}" if event.winner_score is not None else "?"
            tail = f"winner={event.winner_model} (score={score})"
        else:
            tail = event.mode
        print(f"→ synthesized [{event.mode}]: {tail}", file=sys.stderr, flush=True)


class JSONLSink:
    """Append each event as a single JSON line to a file."""

    def __init__(self, path: str):
        self._path = path

    def __call__(self, event: RouterEvent) -> None:
        try:
            payload: dict[str, Any] = {"event": type(event).__name__}
            for k, v in event.__dict__.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    payload[k] = v
                elif isinstance(v, list):
                    payload[k] = list(v)
            with open(self._path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except OSError as exc:
            logger.warning("JSONL sink failed: %s", exc)
