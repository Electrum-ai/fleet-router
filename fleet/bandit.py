"""Outcome-driven Thompson-sampling bandit for (tag, model) selection.

State is a Beta(α, β) posterior per (tag, model). On each sample we draw
one number per arm and pick argmax. On each outcome we update α/β with the
observed reward (mapped to {0, 1}). Persists to JSON if a state_path is set.

Reward signal is the verifier/judge score from the synthesis pipeline —
NOT latency or cost. The bandit learns "which model produces the best
answer for this tag" over time.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ThompsonBandit:
    """Per-(tag, model) Beta-Bernoulli bandit. Thread-safe under file lock."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        # Expand ~ so configs like "~/.fleet/bandit.json" Just Work.
        self._state_path = os.path.expanduser(state_path) if state_path else None
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._state: dict[str, dict[str, list[float]]] = {}
        self._lock = threading.Lock()
        self._load()

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
        partial alpha/beta updates."""
        reward = max(0.0, min(1.0, float(reward)))
        with self._lock:
            tag_state = self._state.setdefault(tag, {})
            ab = tag_state.get(model)
            if ab is None:
                ab = [self._prior_alpha, self._prior_beta]
                tag_state[model] = ab
            ab[0] += reward
            ab[1] += 1.0 - reward
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
        if not self._state_path:
            return
        try:
            tmp_path = self._state_path + ".tmp"
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(self.snapshot(), f, indent=2)
            os.replace(tmp_path, self._state_path)
        except OSError as exc:
            logger.warning("bandit state save failed: %s", exc)
