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
            )
        if result.winner.score < self._abstention_threshold:
            return VerificationResult(
                winner=None, all_scored=result.all_scored,
                rationale=(
                    f"winner score {result.winner.score:.2f} below threshold "
                    f"{self._abstention_threshold}; abstaining"
                ),
                abstain=True,
            )
        return result
