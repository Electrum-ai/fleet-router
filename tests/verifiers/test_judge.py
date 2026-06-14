from unittest.mock import AsyncMock

import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.judge import (
    JudgeVerifier,
    _candidate_label,
    _extract_json,
    _MAX_CANDIDATE_CHARS,
)


def test_extract_json_pure_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded_in_prose():
    text = "Here is the result:\n\n{\"best\": \"A\", \"scores\": {\"A\": 8}}\n\nThanks!"
    assert _extract_json(text) == {"best": "A", "scores": {"A": 8}}


def test_extract_json_returns_none_when_no_json():
    assert _extract_json("just text") is None


def test_extract_json_handles_nested():
    text = '{"outer": {"inner": [1, 2, 3]}}'
    assert _extract_json(text) == {"outer": {"inner": [1, 2, 3]}}


@pytest.mark.asyncio
async def test_judge_picks_best_and_normalizes_scores():
    # swap_order=False isolates single-pass scoring: this test asserts raw
    # normalization (8/10 -> 0.8). With swap-order ON (the default) the
    # order-blind stub returns the same JSON for both passes, so label A
    # scores 8 forward and the reversed-position candidate scores 8 too —
    # averaging yields a 0.55 tie that doesn't exercise normalization.
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 8, "B": 3}, "best": "A", "rationale": "A is better"}'
    ])
    v = JudgeVerifier(provider=provider, judge_model="judge", tag="general", swap_order=False)
    candidates = [
        Candidate("model-a", 0, "answer A"),
        Candidate("model-b", 0, "answer B"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "model-a"
    assert result.winner.score == 0.8
    assert "A is better" in result.rationale


@pytest.mark.asyncio
async def test_judge_falls_back_to_first_candidate_on_unparseable_output():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=["lol the model just emitted prose"])
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [
        Candidate("a", 0, "first"),
        Candidate("b", 0, "second"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"
    assert "unparseable" in result.rationale


@pytest.mark.asyncio
async def test_judge_falls_back_when_provider_returns_nothing():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[None])
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "a"


@pytest.mark.asyncio
async def test_judge_handles_unknown_best_label():
    """When the judge points at a label that doesn't exist, fall back to
    highest-scored. swap_order=False isolates the single-pass best-label
    fallback (the order-blind stub would otherwise average to a tie)."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 4, "B": 9}, "best": "Z"}'
    ])
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=False)
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.model == "b"  # highest score


@pytest.mark.asyncio
async def test_judge_passes_single_candidate_through():
    provider = AsyncMock()
    v = JudgeVerifier(provider=provider, judge_model="judge")
    result = await v.aggregate("p", [Candidate("a", 0, "only")])
    assert result.winner is not None
    assert result.winner.model == "a"
    # No judge call needed for a single candidate.
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_judge_handles_provider_exception():
    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("boom"))
    v = JudgeVerifier(provider=provider, judge_model="judge")
    candidates = [Candidate("a", 0, "x"), Candidate("b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None  # graceful fallback


# --------------------------------------------------------------------------- #
# A — self-preference: delegate to a neutral judge                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_neutral_judge_used_when_judge_is_among_candidates():
    """The configured judge's own model is a candidate AND a neutral model
    exists → aggregate delegates to the neutral judge. Assert the NEUTRAL
    provider did the grading, not the judge's own provider."""
    own_provider = AsyncMock()
    own_provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 9, "B": 1}, "best": "A"}'
    ])
    neutral_provider = AsyncMock()
    neutral_provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 2, "B": 8}, "best": "B"}'
    ])

    def neutral_factory(candidate_models):
        assert "judge-model" in candidate_models
        # Neutral judge bound to a NON-candidate model.
        return JudgeVerifier(
            provider=neutral_provider, judge_model="neutral",
            model_key="neutral", swap_order=False,
        )

    v = JudgeVerifier(
        provider=own_provider, judge_model="judge-model",
        model_key="judge-model", neutral_factory=neutral_factory,
        swap_order=False,
    )
    candidates = [Candidate("judge-model", 0, "mine"), Candidate("other", 0, "theirs")]
    result = await v.aggregate("p", candidates)

    # The neutral judge graded — its verdict (B wins) is what we see.
    neutral_provider.generate.assert_awaited()
    own_provider.generate.assert_not_called()
    assert result.winner.model == "other"
    assert result.winner.score == 0.8


@pytest.mark.asyncio
async def test_no_neutral_available_judges_as_is_with_noted_rationale():
    """Judge is a candidate but neutral_factory returns None (no neutral model)
    → judge as-is, but flag the self-preference risk in the rationale."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 9, "B": 1}, "best": "A"}'
    ])
    v = JudgeVerifier(
        provider=provider, judge_model="judge-model", model_key="judge-model",
        neutral_factory=lambda models: None, swap_order=False,
    )
    candidates = [Candidate("judge-model", 0, "mine"), Candidate("other", 0, "theirs")]
    result = await v.aggregate("p", candidates)

    provider.generate.assert_awaited()  # judged as-is
    assert result.winner.model == "judge-model"
    assert "self-preference" in result.rationale


@pytest.mark.parametrize("swap_order", [True, False])
@pytest.mark.asyncio
async def test_self_pref_no_neutral_multi_model_is_unreliable(swap_order):
    """Case (a): judge IS a candidate, NO neutral available, and 2+ distinct
    candidate models are being compared → the cross-model scores are
    self-preference-biased, so scores_reliable is False (bandit must skip)
    while the self-pref note stays in the rationale. Holds for swap_order
    both True and False (position de-biasing doesn't remove self-pref bias)."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 9, "B": 1}, "best": "A"}'
    ])
    v = JudgeVerifier(
        provider=provider, judge_model="judge-model", model_key="judge-model",
        neutral_factory=lambda models: None, swap_order=swap_order,
    )
    candidates = [
        Candidate("judge-model", 0, "mine"),
        Candidate("other", 0, "theirs"),
    ]
    result = await v.aggregate("p", candidates)

    provider.generate.assert_awaited()  # judged as-is (no neutral)
    assert result.scores_reliable is False
    assert "self-preference" in result.rationale


@pytest.mark.parametrize("swap_order", [True, False])
@pytest.mark.asyncio
async def test_self_pref_no_neutral_single_model_stays_reliable(swap_order):
    """Case (b): judge IS a candidate, NO neutral, but ALL candidates are the
    judge's OWN single model (single-pool self-consistency). There is no
    between-model preference to bias and the bandit has one arm for the tag,
    so scores stay reliable — the note is still present for transparency."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 8, "B": 3}, "best": "A"}'
    ])
    v = JudgeVerifier(
        provider=provider, judge_model="judge-model", model_key="judge-model",
        neutral_factory=lambda models: None, swap_order=swap_order,
    )
    candidates = [
        Candidate("judge-model", 0, "sample one"),
        Candidate("judge-model", 1, "sample two"),
    ]
    result = await v.aggregate("p", candidates)

    provider.generate.assert_awaited()
    assert result.scores_reliable is True
    assert "self-preference" in result.rationale


@pytest.mark.asyncio
async def test_self_pref_with_neutral_is_reliable_and_unnoted():
    """Case (c): a neutral judge IS available → aggregate delegates to it. The
    neutral graded the slate, so scores_reliable is True and there is NO
    self-preference note (the bias was actually removed, not just flagged)."""
    own_provider = AsyncMock()
    own_provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 9, "B": 1}, "best": "A"}'
    ])
    neutral_provider = AsyncMock()
    neutral_provider.generate = AsyncMock(return_value=[
        '{"scores": {"A": 3, "B": 7}, "best": "B"}'
    ])

    def neutral_factory(models):
        return JudgeVerifier(
            provider=neutral_provider, judge_model="neutral",
            model_key="neutral", swap_order=False,
        )

    v = JudgeVerifier(
        provider=own_provider, judge_model="judge-model", model_key="judge-model",
        neutral_factory=neutral_factory, swap_order=False,
    )
    candidates = [
        Candidate("judge-model", 0, "mine"),
        Candidate("other", 0, "theirs"),
    ]
    result = await v.aggregate("p", candidates)

    own_provider.generate.assert_not_called()
    neutral_provider.generate.assert_awaited()
    assert result.scores_reliable is True
    assert "self-preference" not in result.rationale
    assert result.winner.model == "other"  # neutral's verdict


@pytest.mark.asyncio
async def test_no_self_preference_when_judge_not_a_candidate():
    """Judge model is NOT among candidates → no delegation, no note."""
    neutral_called = False

    def neutral_factory(models):
        nonlocal neutral_called
        neutral_called = True
        return None

    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=['{"scores": {"A": 7, "B": 4}, "best": "A"}'])
    v = JudgeVerifier(
        provider=provider, judge_model="outside-judge", model_key="outside-judge",
        neutral_factory=neutral_factory, swap_order=False,
    )
    result = await v.aggregate("p", [Candidate("a", 0, "x"), Candidate("b", 0, "y")])
    assert not neutral_called
    assert "self-preference" not in result.rationale


# --------------------------------------------------------------------------- #
# B — position bias: swap-order averaging                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_swap_order_runs_judge_twice_and_averages():
    """swap_order=True → two judge calls; per-candidate scores are averaged.
    The order-aware stub favors whichever candidate sits in position A, so the
    averaged scores land between the two passes."""
    calls = []

    async def fake_generate(req):
        calls.append(req.prompt)
        # Position A always scores 8, position B scores 2 (a pure position bias).
        return ['{"scores": {"A": 8, "B": 2}, "best": "A"}']

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=fake_generate)
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=True)
    candidates = [Candidate("model-a", 0, "answer A"), Candidate("model-b", 0, "answer B")]
    result = await v.aggregate("p", candidates)

    assert provider.generate.await_count == 2
    # Forward: A=model-a=0.8, B=model-b=0.2. Reversed: A=model-b=0.8, B=model-a=0.2.
    # Averaged: both 0.5 — position bias fully cancelled.
    scores = {c.model: c.score for c in result.all_scored}
    assert scores["model-a"] == pytest.approx(0.5)
    assert scores["model-b"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_swap_order_disagreement_is_surfaced():
    """When forward and reversed passes name different winners, the
    disagreement is recorded in the rationale (low-confidence signal)."""
    async def fake_generate(req):
        # Always pick position A as best → forward best=model-a, reversed best=model-b.
        return ['{"scores": {"A": 7, "B": 6}, "best": "A"}']

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=fake_generate)
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=True)
    candidates = [Candidate("model-a", 0, "x"), Candidate("model-b", 0, "y")]
    result = await v.aggregate("p", candidates)

    assert "disagreement" in result.rationale
    # Averaged scores are still reliable signal (both passes parsed fine).
    assert result.scores_reliable is True


@pytest.mark.asyncio
async def test_swap_order_off_runs_single_pass():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=['{"scores": {"A": 8, "B": 3}, "best": "A"}'])
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=False)
    candidates = [Candidate("model-a", 0, "x"), Candidate("model-b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert provider.generate.await_count == 1
    assert result.winner.model == "model-a"
    assert result.winner.score == 0.8


@pytest.mark.asyncio
async def test_swap_pass_failure_degrades_to_single_pass():
    """If the reversed pass fails but the forward pass succeeded, keep the
    forward scores rather than discarding everything."""
    outputs = ['{"scores": {"A": 8, "B": 3}, "best": "A"}', None]

    async def fake_generate(req):
        return [outputs.pop(0)]

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=fake_generate)
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=True)
    candidates = [Candidate("model-a", 0, "x"), Candidate("model-b", 0, "y")]
    result = await v.aggregate("p", candidates)
    assert provider.generate.await_count == 2
    assert result.winner.model == "model-a"
    assert result.winner.score == 0.8
    assert "single-pass" in result.rationale


# --------------------------------------------------------------------------- #
# C — slate hygiene: truncation + >26-candidate labels                        #
# --------------------------------------------------------------------------- #


def test_candidate_label_no_collision_past_26():
    labels = [_candidate_label(i) for i in range(60)]
    assert len(set(labels)) == 60          # no collisions
    assert labels[:3] == ["A", "B", "C"]
    assert labels[25] == "Z"
    assert labels[26] == "AA"
    assert labels[27] == "AB"
    # The old chr(65+i) scheme produced these for i in 26..28 — assert we don't.
    assert "[" not in labels and "\\" not in labels and "]" not in labels


@pytest.mark.asyncio
async def test_per_candidate_text_truncated_in_prompt():
    """A long candidate is truncated in the judge prompt, but the FULL text is
    preserved on the returned winner."""
    captured = {}

    async def fake_generate(req):
        captured["prompt"] = req.prompt
        return ['{"scores": {"A": 9, "B": 1}, "best": "A"}']

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=fake_generate)
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=False)
    long_text = "x" * (_MAX_CANDIDATE_CHARS + 5000)
    candidates = [Candidate("model-a", 0, long_text), Candidate("model-b", 0, "short")]
    result = await v.aggregate("p", candidates)

    assert "[truncated]" in captured["prompt"]
    # The prompt never carried the full blob.
    assert long_text not in captured["prompt"]
    # The winner keeps the full, untruncated candidate text.
    assert result.winner.model == "model-a"
    assert result.winner.text == long_text


@pytest.mark.asyncio
async def test_27_plus_candidates_label_and_winner_map_correctly():
    """27 candidates → labels must not collide and the named winner must map
    back to the right candidate (regression for chr(65+i) overflow)."""
    captured = {}

    async def fake_generate(req):
        captured["prompt"] = req.prompt
        # Pick label "AA" (the 27th candidate, index 26) as best.
        scores = {_candidate_label(i): 1 for i in range(27)}
        scores["AA"] = 9
        import json as _json
        return [_json.dumps({"scores": scores, "best": "AA"})]

    provider = AsyncMock()
    provider.generate = AsyncMock(side_effect=fake_generate)
    v = JudgeVerifier(provider=provider, judge_model="judge", swap_order=False)
    candidates = [Candidate(f"model-{i}", 0, f"answer {i}") for i in range(27)]
    result = await v.aggregate("p", candidates)

    # Label AA is the 27th candidate (index 26) → model-26.
    assert result.winner.model == "model-26"
    assert result.winner.score == 0.9
    # Prompt used the safe labels, no overflow characters.
    assert "Candidate AA" in captured["prompt"]
    assert "Candidate [" not in captured["prompt"]
