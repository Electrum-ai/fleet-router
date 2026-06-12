"""VerifierSynthesizer — replaces heuristic Synthesizer with verifier-driven
selection that supports self-consistency (multiple samples per model)."""
from __future__ import annotations

from typing import Optional

from fleet.text import strip_thinking
from fleet.verifiers.base import (
    DEFAULT_ABSTENTION_THRESHOLD,
    Candidate,
    VerificationResult,
)
from fleet.verifiers.registry import VerifierRegistry


class VerifierSynthesizer:
    """Async synthesizer that delegates to tag-specific Verifiers.

    Input shape is `dict[model, list[str]]` so self-consistency
    (samples_per_model > 1) works natively — every sample becomes a
    candidate.
    """

    def __init__(
        self,
        registry: Optional[VerifierRegistry] = None,
        abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
    ):
        self._registry = registry or VerifierRegistry()
        self._abstention_threshold = abstention_threshold

    def verifier_for(self, task_tag: str):
        """Resolve the Verifier that will score `task_tag` (post-fallback).
        Exposed so the router can tell a discriminative verifier (judge /
        executable / math) from the order-dependent HeuristicVerifier when
        gating refinement accepts."""
        return self._registry.for_tag(task_tag)

    async def pick(
        self,
        prompt: str,
        samples_per_model: dict[str, list[str]],
        task_tag: str,
    ) -> VerificationResult:
        candidates: list[Candidate] = []
        for model, samples in samples_per_model.items():
            for i, text in enumerate(samples):
                if not text or not text.strip():
                    continue
                # Strip chain-of-thought exactly ONCE, here at the candidate
                # boundary, so scoring, judge/escalation prompts, abstention
                # summaries, refinement input, and the returned winner are all
                # consistently clean. Keep the original in raw_text. A sample
                # that is *only* a <think> block collapses to "" and is
                # dropped — it never masqueraded as a real answer.
                cleaned = strip_thinking(text)
                if not cleaned:
                    continue
                candidates.append(
                    Candidate(
                        model=model, sample_idx=i, text=cleaned, raw_text=text
                    )
                )

        if not candidates:
            return VerificationResult(
                winner=None, all_scored=[],
                rationale="all models failed", abstain=True,
            )

        verifier = self._registry.for_tag(task_tag)
        result = await verifier.aggregate(prompt, candidates)

        return self._apply_abstention(result)

    async def score_candidates(
        self,
        prompt: str,
        candidates: list[Candidate],
        task_tag: str,
    ) -> VerificationResult:
        """Score an explicit, already-clean list of Candidates with the tag
        verifier — no sample flattening, no chain-of-thought stripping, and no
        calibrated-abstention overlay. Used by the router to close the loop on
        refinement/escalation outputs: it returns the raw verifier scores
        (winner + all_scored) so the caller can compare an ad-hoc rewrite
        against the originals under the SAME verifier the round used.

        Callers are responsible for passing strip_thinking()'d text.
        """
        if not candidates:
            return VerificationResult(
                winner=None, all_scored=[],
                rationale="no candidates", abstain=True,
            )
        verifier = self._registry.for_tag(task_tag)
        return await verifier.aggregate(prompt, candidates)

    def _apply_abstention(
        self, result: VerificationResult
    ) -> VerificationResult:
        # Calibrated abstention: even if verifier picked a winner, abstain
        # when the winner's score is below threshold. Verifier-set
        # `abstain=True` always wins (verifier knows best).
        if result.abstain:
            return result
        if result.winner is None:
            return VerificationResult(
                winner=None, all_scored=result.all_scored,
                rationale=result.rationale or "verifier returned no winner",
                abstain=True,
                # Preserve reliability — an unreliable result must stay
                # unreliable through the rebuild so the bandit (which runs
                # even on abstain) never ingests judge-failure noise.
                scores_reliable=result.scores_reliable,
            )
        if result.winner.score < self._abstention_threshold:
            return VerificationResult(
                winner=None, all_scored=result.all_scored,
                rationale=(
                    f"winner score {result.winner.score:.2f} below threshold "
                    f"{self._abstention_threshold}; abstaining"
                ),
                abstain=True,
                scores_reliable=result.scores_reliable,
            )
        return result
