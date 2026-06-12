import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.heuristic import HeuristicVerifier


@pytest.mark.asyncio
async def test_heuristic_picks_via_underlying_synthesizer():
    v = HeuristicVerifier(tag="code")
    candidates = [
        Candidate("a", 0, "broken syntax ("),
        Candidate("b", 0, "def foo():\n    return 1"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "def foo" in result.winner.text


@pytest.mark.asyncio
async def test_heuristic_abstains_on_synthesizer_tie():
    """When the heuristic synthesizer returns a dict (tie), wrap as abstention."""
    v = HeuristicVerifier(tag="general")
    candidates = [
        Candidate("a", 0, "abc"),  # length 3
        Candidate("b", 0, "def"),  # length 3 — tie
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_heuristic_handles_empty_candidates():
    v = HeuristicVerifier()
    result = await v.aggregate("p", [])
    assert result.abstain


@pytest.mark.asyncio
async def test_heuristic_winner_attribution_survives_trailing_newline():
    """Regression: Synthesizer.pick normalizes (strips) candidate text, so a
    raw candidate with a trailing newline never matched `c.text == chosen`.
    The real winner fell through to scored[0] @ 0.3 ("heuristic non-winner"),
    which is below the 0.4 abstention threshold — making the router abstain on
    essentially every multi-candidate prompt with no judge configured. The
    winner must get its real 0.7 score, not 0.3."""
    v = HeuristicVerifier(tag="general")
    candidates = [
        Candidate("a", 0, "this is a much longer response with many words here\n"),
        Candidate("b", 0, "short\n"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"
    assert result.winner.score == 0.7
    assert result.winner.score != 0.3
    assert not result.abstain


@pytest.mark.asyncio
async def test_heuristic_winner_attribution_survives_thinking_block():
    """Same attribution path, but the raw candidate carries a <think> block
    that Synthesizer.pick strips. The stripped winner must still be matched
    back to its candidate and scored 0.7."""
    v = HeuristicVerifier(tag="general")
    candidates = [
        Candidate("a", 0, "<think>let me reason</think>this is the much longer winning response"),
        Candidate("b", 0, "tiny"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"
    assert result.winner.score == 0.7
