import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.math import MathVerifier, _extract_final_answer


def test_extract_final_answer_explicit_marker():
    assert _extract_final_answer("So the answer is 42.") == "42"
    assert _extract_final_answer("Final answer: 3.14") == "3.14"
    assert _extract_final_answer(r"\boxed{17}") == "17"


def test_extract_final_answer_falls_back_to_last_number():
    text = "First we compute 10, then 20, then 30. So 30."
    assert _extract_final_answer(text) == "30"


def test_extract_final_answer_handles_decimals_and_negatives():
    assert _extract_final_answer("Answer: -2.5") == "-2.5"
    assert _extract_final_answer("Answer: 1.5e3") == "1500"


def test_extract_final_answer_normalizes():
    """42 and 42.0 should be considered the same numeric answer."""
    assert _extract_final_answer("Answer: 42.0") == "42"


def test_extract_final_answer_returns_none_on_no_number():
    assert _extract_final_answer("I don't know.") is None


def test_extract_final_answer_marker_allows_is():
    """Regression: "the answer is 42" was not matched by the marker regex (no
    allowance for the word "is") and silently fell to the last-number
    fallback. The marker path must capture it directly."""
    assert _extract_final_answer("the answer is 42") == "42"
    assert _extract_final_answer("So the final answer is 7.") == "7"


def test_extract_final_answer_comma_grouped_integer():
    """Regression: comma grouping disagreed between the two paths — the marker
    path captured only '1,234' and the fallback split into ['1','234','567']
    yielding '567'. Both must now agree on the full number."""
    assert _extract_final_answer("The answer is 1,234,567") == "1234567"
    assert _extract_final_answer("answer is 1,234,567") == "1234567"
    # Fallback path (no marker) must also handle the grouping.
    assert _extract_final_answer("after a long computation we get 1,234,567") == "1234567"


def test_extract_final_answer_simple_fraction():
    """Regression: "x = 1/2" extracted '1' (numerator only). It must evaluate
    to a normalized decimal so it votes as the same value as candidates that
    write 0.5 — not corrupt the tally with a wrong integer."""
    assert _extract_final_answer("x = 1/2") == "0.5"
    assert _extract_final_answer("the answer is 3/4") == "0.75"


def test_to_number_rejects_zero_denominator_fraction():
    """A divide-by-zero fraction is undefined — the evaluator declines it
    (returns None) rather than voting the numerator or denominator."""
    from fleet.verifiers.math import _to_number

    assert _to_number("1/0") is None
    assert _to_number("1/2") == "0.5"
    assert _to_number("1,234") == "1234"


def test_extract_final_answer_zero_denominator_does_not_leak_via_fallback():
    """Regression: `_to_number('1/0')` correctly returned None, but the
    last-number fallback re-scanned the whole string and `_NUMBER_RE` tokenized
    "1/0" into ['1','0'], returning the bare denominator '0'. The fallback is
    now fraction-aware and skips tokens `_to_number` rejects, so a malformed
    zero-denominator fraction does NOT vote — matching the `_to_number`
    docstring contract on BOTH the helper and the fallback paths."""
    assert _extract_final_answer("the answer is 1/0") is None
    assert _extract_final_answer("x = 3/0") is None
    # Bare phrasing (no marker → fallback path) leaks too without the fix.
    assert _extract_final_answer("after dividing we get 5/0") is None


def test_extract_final_answer_bare_fraction_agrees_with_marker_phrasing():
    """Regression: the marker path evaluated "x = 1/2" -> '0.5' while the
    fallback tokenized "we compute 1/2 here" into ['1','2'] -> '2'. The two
    paths now agree because the fallback parses the fraction as one token."""
    assert _extract_final_answer("x = 1/2") == "0.5"          # marker path
    assert _extract_final_answer("we compute 1/2 here") == "0.5"  # fallback path
    assert _extract_final_answer("x = 3/4") == _extract_final_answer("roughly 3/4")


def test_extract_final_answer_trailing_period():
    assert _extract_final_answer("= 42.") == "42"


def test_extract_final_answer_boxed_and_scientific_still_work():
    assert _extract_final_answer(r"\boxed{42}") == "42"
    assert _extract_final_answer("Answer: -3.5e2") == "-350"


@pytest.mark.asyncio
async def test_math_verifier_fraction_and_decimal_vote_together():
    """A candidate writing 1/2 and one writing 0.5 must land in the same
    vote bucket — majority voting stays coherent across notations."""
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "x = 1/2"),
        Candidate("b", 0, "The answer is 0.5"),
        Candidate("c", 0, "I get 0.9"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert not result.abstain
    # a and b agree (0.5) → majority of 2/3.
    assert "0.5" in result.rationale


@pytest.mark.asyncio
async def test_math_verifier_majority_vote_picks_winner():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "Answer: 42"),
        Candidate("b", 0, "I think the answer is 42."),
        Candidate("c", 0, "The answer is 99."),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "42" in result.winner.text
    assert not result.abstain


@pytest.mark.asyncio
async def test_math_verifier_abstains_on_no_majority():
    """3 candidates, 3 different answers → no majority → abstain."""
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "Answer: 1"),
        Candidate("b", 0, "Answer: 2"),
        Candidate("c", 0, "Answer: 3"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_math_verifier_abstains_on_no_extractable_numbers():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "I don't know"),
        Candidate("b", 0, "no idea"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain


@pytest.mark.asyncio
async def test_math_verifier_strips_thinking_before_extract():
    v = MathVerifier()
    candidates = [
        Candidate("a", 0, "<think>let me work this out: 1+1=2, no wait, 2+2=4</think>The answer is 7."),
        Candidate("b", 0, "Answer: 7"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    # Both should agree on 7 — thinking tokens shouldn't pollute extraction.
    assert "7" in result.winner.text
