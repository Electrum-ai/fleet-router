"""Verifier Protocol and shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Candidate:
    """A single sampled response from one model.

    ``text`` is the chain-of-thought-stripped final answer — what scoring,
    judge/escalation prompts, abstention summaries, and the returned winner all
    consume. ``raw_text`` preserves the original generation (including any
    ``<think>`` block) for the rare consumer that needs it.
    """
    model: str
    sample_idx: int
    text: str
    score: float = 0.0
    notes: str = ""
    raw_text: str = ""

    def with_score(self, score: float, notes: str = "") -> "Candidate":
        return replace(self, score=score, notes=notes or self.notes)


@dataclass
class VerificationResult:
    """Outcome of running a verifier across all candidates for one prompt.

    `abstain=True` signals "no candidate is good enough — return uncertainty
    structure instead of guessing." Callers (router) can choose to honor it
    or escalate to a stronger model with all candidates as context.

    `scores_reliable=False` means the verifier returned scores but they're
    fallback/error values, not real quality signals — typically when the
    judge model crashed, returned empty, or produced unparseable output.
    Bandit must NOT update from unreliable scores or it will poison its
    posteriors with judge-failure noise.
    """
    winner: Optional[Candidate]
    all_scored: list[Candidate]
    rationale: str = ""
    abstain: bool = False
    scores_reliable: bool = True

    @property
    def winner_text(self) -> Optional[str]:
        return self.winner.text if self.winner else None


@runtime_checkable
class Verifier(Protocol):
    """Async tag-specific scorer + selector."""

    tag: str

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        ...


# Threshold below which `winner.score` triggers calibrated abstention.
DEFAULT_ABSTENTION_THRESHOLD = 0.4
