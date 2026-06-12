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


# --------------------------------------------------------------------------- #
# Per-tag abstention thresholds (incommensurable-scales fix)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_per_tag_threshold_abstains_where_default_would_accept():
    """A per-tag threshold of 0.6 makes a 0.5 winner abstain on that tag, even
    though the global default of 0.4 would have accepted it."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("math", winner_idx=0, scores=[0.5, 0.3]))
    s = VerifierSynthesizer(
        reg,
        abstention_threshold=0.4,
        abstention_thresholds={"math": 0.6, "code": 0.55},
    )
    result = await s.pick("p", {"a": ["x"], "b": ["y"]}, "math")
    assert result.abstain
    assert result.winner is None


@pytest.mark.asyncio
async def test_per_tag_threshold_unlisted_tag_uses_default():
    """A tag absent from the override map uses the global default unchanged: a
    0.5 winner clears the 0.4 default and is accepted."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("reasoning", winner_idx=0, scores=[0.5, 0.3]))
    s = VerifierSynthesizer(
        reg,
        abstention_threshold=0.4,
        abstention_thresholds={"math": 0.6},  # reasoning not listed
    )
    result = await s.pick("p", {"a": ["x"], "b": ["y"]}, "reasoning")
    assert not result.abstain
    assert result.winner is not None
    assert result.winner.score == 0.5


@pytest.mark.asyncio
async def test_empty_override_map_reproduces_default_behavior():
    """Empty override map ⇒ every tag uses the global default exactly as
    before (regression guard for the default path)."""
    reg = VerifierRegistry()
    reg.register(_FixedScoreVerifier("math", winner_idx=0, scores=[0.5]))
    s = VerifierSynthesizer(reg, abstention_threshold=0.4, abstention_thresholds={})
    result = await s.pick("p", {"a": ["x"]}, "math")
    assert not result.abstain
    assert result.winner is not None


def test_abstention_threshold_for_resolves_override_then_default():
    s = VerifierSynthesizer(
        VerifierRegistry(),
        abstention_threshold=0.4,
        abstention_thresholds={"math": 0.6},
    )
    assert s.abstention_threshold_for("math") == 0.6
    assert s.abstention_threshold_for("code") == 0.4   # falls back
    assert s.abstention_threshold_for(None) == 0.4     # direct/no-tag path


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


def test_apply_abstention_preserves_scores_reliable():
    """F7: an unreliable result that trips the abstention threshold must stay
    unreliable through the rebuild. _update_bandit runs even on abstain, so a
    reset-to-True here would leak judge-failure noise into the posteriors."""
    s = VerifierSynthesizer(VerifierRegistry(), abstention_threshold=0.4)

    # Below-threshold winner branch.
    low = Candidate("m", 0, "x", score=0.2)
    unreliable = VerificationResult(
        winner=low, all_scored=[low], rationale="judge unavailable",
        scores_reliable=False,
    )
    out = s._apply_abstention(unreliable)
    assert out.abstain is True
    assert out.winner is None
    assert out.scores_reliable is False

    # winner-is-None branch.
    none_winner = VerificationResult(
        winner=None, all_scored=[low], rationale="no winner",
        scores_reliable=False,
    )
    out2 = s._apply_abstention(none_winner)
    assert out2.abstain is True
    assert out2.scores_reliable is False

    # A reliable result that abstains stays reliable (no regression).
    reliable = VerificationResult(
        winner=low, all_scored=[low], rationale="low", scores_reliable=True,
    )
    out3 = s._apply_abstention(reliable)
    assert out3.abstain is True
    assert out3.scores_reliable is True


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
