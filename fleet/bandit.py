"""Outcome-driven Thompson-sampling bandit for (tag, model) selection.

State is a Beta(α, β) posterior per (tag, model). On each sample we draw
one number per arm and pick argmax. On each outcome we update α/β with the
observed reward (mapped to {0, 1}). Persists to JSON if a state_path is set.

Reward signal is the verifier/judge score from the synthesis pipeline —
NOT latency or cost. The bandit learns "which model produces the best
answer for this tag" over time.

Persistence is debounced (one disk write per `save_every` updates) and
serialized through a dedicated save lock so concurrent `update()` calls
on the proxy's event loop don't race on the `.tmp` rename. A final
flush runs at process exit so debounced state isn't lost.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import random
import threading
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class ThompsonBandit:
    """Per-(tag, model) Beta-Bernoulli bandit. Thread-safe under file lock."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        save_every: int = 25,
    ):
        # Expand ~ so configs like "~/.fleet/bandit.json" Just Work.
        self._state_path = os.path.expanduser(state_path) if state_path else None
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._state: dict[str, dict[str, list[float]]] = {}
        self._lock = threading.Lock()
        # Separate lock so the (potentially slow) disk write doesn't block
        # readers/writers of the state dict — but is still serialized so
        # two concurrent saves can't clobber each other on the .tmp rename.
        self._save_lock = threading.Lock()
        self._save_every = max(1, int(save_every))
        self._pending_writes = 0
        self._load()
        # Final flush at shutdown — debounced updates would otherwise be
        # lost. Bound method ref keeps the bandit alive long enough.
        if self._state_path:
            atexit.register(self.flush)

    def _key(self, tag: str, model: str) -> tuple[str, str]:
        return tag, model

    def _params(self, tag: str, model: str) -> tuple[float, float]:
        with self._lock:
            tag_state = self._state.setdefault(tag, {})
            ab = tag_state.get(model)
            if ab is None:
                ab = [self._prior_alpha, self._prior_beta]
                tag_state[model] = ab
            return ab[0], ab[1]

    def select(self, tag: str, models: list[str]) -> Optional[str]:
        """Sample one Beta per model; return argmax. Returns None if `models`
        is empty."""
        if not models:
            return None
        best_model = models[0]
        best_draw = -1.0
        for m in models:
            a, b = self._params(tag, m)
            draw = random.betavariate(a, b)
            if draw > best_draw:
                best_draw = draw
                best_model = m
        return best_model

    def rank(self, tag: str, models: list[str]) -> list[str]:
        """Return models sorted by Thompson draw (descending)."""
        if not models:
            return []
        draws: list[tuple[float, str]] = []
        for m in models:
            a, b = self._params(tag, m)
            draws.append((random.betavariate(a, b), m))
        draws.sort(key=lambda x: -x[0])
        return [m for _, m in draws]

    def update(self, tag: str, model: str, reward: float) -> None:
        """Update posterior. Reward ∈ [0, 1]. Treats reward as a Bernoulli
        outcome with that probability — fractional rewards split into
        partial alpha/beta updates. Writes are debounced — see save_every."""
        reward = max(0.0, min(1.0, float(reward)))
        should_save = False
        with self._lock:
            tag_state = self._state.setdefault(tag, {})
            ab = tag_state.get(model)
            if ab is None:
                ab = [self._prior_alpha, self._prior_beta]
                tag_state[model] = ab
            ab[0] += reward
            ab[1] += 1.0 - reward
            self._pending_writes += 1
            if self._pending_writes >= self._save_every:
                self._pending_writes = 0
                should_save = True
        if should_save:
            self._save()

    def flush(self) -> None:
        """Force any debounced writes to disk. Called from atexit and
        available for callers that need a synchronization point."""
        with self._lock:
            self._pending_writes = 0
        self._save()

    def posterior_mean(self, tag: str, model: str) -> float:
        a, b = self._params(tag, model)
        return a / (a + b)

    def snapshot(self) -> dict[str, dict[str, list[float]]]:
        with self._lock:
            return {tag: {m: list(ab) for m, ab in models.items()}
                    for tag, models in self._state.items()}

    def _load(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return
            for tag, models in raw.items():
                if not isinstance(models, dict):
                    continue
                self._state[str(tag)] = {
                    str(m): [float(ab[0]), float(ab[1])]
                    for m, ab in models.items()
                    if isinstance(ab, list) and len(ab) == 2
                }
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("bandit state load failed (%s); starting fresh", exc)

    def _save(self) -> None:
        """Atomic, race-free disk save. The save lock serializes concurrent
        writers; the unique tmp suffix means even pre-existing stale .tmp
        files from a prior crash never collide with the current write."""
        if not self._state_path:
            return
        # Snapshot under the state lock — short critical section, then
        # release before touching disk so updaters aren't blocked.
        snapshot = self.snapshot()
        with self._save_lock:
            try:
                # Unique tmp path per save call — defense in depth on top
                # of the save_lock so a SIGKILL'd half-write can't get
                # promoted to the canonical path by a later os.replace.
                tmp_path = f"{self._state_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
                with open(tmp_path, "w") as f:
                    json.dump(snapshot, f, indent=2)
                os.replace(tmp_path, self._state_path)
            except OSError as exc:
                logger.warning("bandit state save failed: %s", exc)
