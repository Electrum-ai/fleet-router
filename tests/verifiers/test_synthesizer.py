import pytest

from fleet.verifiers.base import Candidate, VerificationResult, Verifier
from fleet.verifiers.registry import VerifierRegistry
from fleet.verifiers.synthesizer import VerifierSynthesizer


class _FixedScoreVerifier:
    def __init__(self, tag, winner_idx, scores):
        self.tag = tag
        self._winner_idx = winner_idx
        self._scores = scores

    async def aggregate(self, prompt, candidates):
        scored = [c.with_score(self._scores[i]) for i, c in enumerate(candidates)]
        winner = scored[self._winner_idx] if self._winner_idx is not None else None
        return VerificationResult(
            winner=winner, all_scored=scored,
            rationale="fixed",
            abstain=winner is None,
        )


@pytest.mark.asyncio
async def test_synthesizer_returns_winner():
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("code", winner_idx=1, scores=[0.5, 0.9]))
    s = VerifierSynthesizer(reg)
    result = await s.pick("p", {"a": ["bad"], "b": ["good"]}, "code")
    assert result.winner is not None
    assert result.winner.text == "good"


@pytest.mark.asyncio
async def test_synthesizer_abstains_below_threshold():
    """Even when the verifier picks a winner, low score triggers abstention."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("code", winner_idx=0, scores=[0.2, 0.1]))
    s = VerifierSynthesizer(reg, abstention_threshold=0.4)
    result = await s.pick("p", {"a": ["x"], "b": ["y"]}, "code")
    assert result.abstain
    assert result.winner is None


@pytest.mark.asyncio
async def test_synthesizer_flattens_samples_per_model():
    """N samples per model become N candidates."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("math", winner_idx=2, scores=[0.5, 0.5, 0.95, 0.5]))
    s = VerifierSynthesizer(reg)
    samples = {
        "model-a": ["sample-a-0", "sample-a-1"],
        "model-b": ["sample-b-0", "sample-b-1"],
    }
    result = await s.pick("p", samples, "math")
    assert result.winner is not None
    # Winner should be the third candidate flattened in iteration order.
    assert "sample-b-0" == result.winner.text


@pytest.mark.asyncio
async def test_synthesizer_filters_empty_samples():
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("math", winner_idx=0, scores=[0.9]))
    s = VerifierSynthesizer(reg)
    result = await s.pick(
        "p",
        {"a": [""], "b": ["   "], "c": ["real answer"]},
        "math",
    )
    assert result.winner is not None
    assert result.winner.text == "real answer"


@pytest.mark.asyncio
async def test_synthesizer_abstains_when_no_valid_candidates():
    reg = VerifierRegistry()
    s = VerifierSynthesizer(reg)
    result = await s.pick("p", {"a": [], "b": [None] if False else []}, "code")
    assert result.abstain


@pytest.mark.asyncio
async def test_synthesizer_uses_heuristic_for_unregistered_tag():
    """When no Verifier is registered for the tag, falls back to HeuristicVerifier."""
    reg = VerifierRegistry()  # nothing registered
    s = VerifierSynthesizer(reg, abstention_threshold=0.0)  # disable abstention
    result = await s.pick(
        "p",
        {"a": ["short"], "b": ["this is a much longer creative response with words"]},
        "creative",
    )
    # HeuristicVerifier wraps the heuristic Synthesizer's pick logic.
    assert result.winner is not None


@pytest.mark.asyncio
async def test_synthesizer_strips_thinking_at_candidate_boundary():
    """Regression (BUG 1): chain-of-thought must be stripped exactly once, at
    the candidate boundary, so Candidate.text (consumed by scoring, prompts,
    abstention dumps, and the returned winner) is always clean. The raw
    generation is preserved in raw_text."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("general", winner_idx=0, scores=[0.9]))
    s = VerifierSynthesizer(reg)
    result = await s.pick(
        "p",
        {"a": ["<thinking>secret reasoning the user must never see</thinking>final answer"]},
        "general",
    )
    assert result.winner is not None
    assert result.winner.text == "final answer"
    assert "<thinking>" not in result.winner.text
    assert "secret reasoning" not in result.winner.text
    # Raw generation retained for any consumer that needs it.
    assert "secret reasoning" in result.winner.raw_text


@pytest.mark.asyncio
async def test_synthesizer_drops_all_thinking_candidate():
    """A sample that is ONLY a <think> block collapses to empty and must not
    masquerade as a real candidate."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("general", winner_idx=0, scores=[0.9]))
    s = VerifierSynthesizer(reg)
    result = await s.pick(
        "p",
        {"a": ["<think>still thinking, never answered</think>"], "b": ["the real answer"]},
        "general",
    )
    assert result.winner is not None
    assert result.winner.text == "the real answer"
    # Only one real candidate survived.
    assert len(result.all_scored) == 1
