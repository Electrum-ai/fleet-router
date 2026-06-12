"""Verifier-mode behavior of FleetRouter (the new default).

Existing heuristic-mode behavior lives in test_router.py — these tests cover
the verifier path: self-consistency dispatch, abstention, escalation, and
refinement.
"""
from unittest.mock import AsyncMock, patch

import pytest

from fleet.config import (
    Config,
    EscalationConfig,
    ModelEntry,
    RefinementConfig,
    SamplingConfig,
    SynthesisConfig,
)
from fleet.router import FleetRouter
from fleet.verifiers.base import Candidate, VerificationResult


@pytest.fixture
def config():
    return Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 3, "default": 1}),
    )


@pytest.fixture
def router(config):
    r = FleetRouter(config)
    r._registry._available = {"model-a", "model-b"}
    r._registry._refreshed = True
    return r


@pytest.mark.asyncio
async def test_verifier_path_uses_run_multi_with_configured_samples(router):
    """sampling.samples_by_tag['math'] = 3 → run_multi(samples=3)."""
    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["the answer is 7"], "model-b": ["the answer is 7"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "the answer is 7", score=0.9),
            all_scored=[],
        )
        result = await router.ask("solve 5+2")
    assert result == "the answer is 7"
    assert mock_multi.call_args.kwargs["samples"] == 3


@pytest.mark.asyncio
async def test_verifier_abstention_returns_calibrated_uncertainty(router):
    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["1"], "model-b": ["2"]}
        mock_pick.return_value = VerificationResult(
            winner=None,
            all_scored=[
                Candidate("model-a", 0, "answer 1", score=0.3),
                Candidate("model-b", 0, "answer 2", score=0.3),
            ],
            rationale="no majority",
            abstain=True,
        )
        result = await router.ask("solve 5+2")
    assert "uncertain" in result
    assert "no majority" in result
    assert "model-a" in result and "model-b" in result


@pytest.mark.asyncio
async def test_escalation_runs_when_verifier_abstains():
    config = Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
            "judge": ModelEntry(tags=["math"], priority=3),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        escalation=EscalationConfig(enabled=True, model="judge", score_threshold=0.6),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b", "judge"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["1"], "model-b": ["2"]}
        mock_pick.return_value = VerificationResult(
            winner=None,
            all_scored=[Candidate("model-a", 0, "1", score=0.2)],
            abstain=True,
        )
        mock_run.return_value = {"judge": "the correct synthesized answer"}
        result = await router.ask("solve")
    assert result == "the correct synthesized answer"
    # Escalation called dispatcher.run with the judge model.
    assert mock_run.call_args[0][1] == ["judge"]


@pytest.mark.asyncio
async def test_refinement_runs_critique_then_revise():
    config = Config(
        models={"model-a": ModelEntry(tags=["general"], priority=1),
                "critic": ModelEntry(tags=["general"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        refinement=RefinementConfig(enabled=True, critique_model="critic"),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "critic"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["draft answer"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "draft answer", score=0.8),
            all_scored=[],
        )
        mock_run.side_effect = [
            {"critic": "you forgot to mention X"},  # critique
            {"critic": "draft answer plus X"},      # revise
        ]
        result = await router.ask("explain something")
    assert result == "draft answer plus X"
    assert mock_run.await_count == 2


@pytest.mark.asyncio
async def test_refinement_skipped_on_no_critique_needed():
    config = Config(
        models={"model-a": ModelEntry(tags=["general"], priority=1),
                "critic": ModelEntry(tags=["general"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        refinement=RefinementConfig(enabled=True, critique_model="critic"),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "critic"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["good draft"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "good draft", score=0.9),
            all_scored=[],
        )
        mock_run.return_value = {"critic": "no critique needed"}
        result = await router.ask("explain")
    # Only the critique call was made; no revise call because critic said it was fine.
    assert result == "good draft"
    assert mock_run.await_count == 1


# ---------- self-judge bias regression guards (audit finding #4) ----------


def test_pick_arbiter_returns_configured_when_not_a_candidate():
    """If the configured arbiter wasn't dispatched as a candidate, no
    rotation needed — return it as-is."""
    config = Config(
        models={"a": ModelEntry(tags=["math"]), "b": ModelEntry(tags=["math"])},
    )
    router = FleetRouter(config)
    router._registry._available = {"a", "b", "judge"}
    router._registry._refreshed = True

    assert router._pick_arbiter("judge", {"a", "b"}) == "judge"


def test_pick_arbiter_rotates_to_neutral_alt_when_configured_is_candidate():
    """If the configured arbiter ALSO appears in the candidate set, swap
    to a different available model. Self-judging is a documented LLM
    bias — judges over-rate their own outputs."""
    config = Config(
        models={
            "a": ModelEntry(tags=["math"]),
            "b": ModelEntry(tags=["math"]),
            "neutral": ModelEntry(tags=["math"]),
        },
    )
    router = FleetRouter(config)
    router._registry._available = {"a", "b", "neutral"}
    router._registry._refreshed = True

    # configured="a" is a candidate; "neutral" is the only non-candidate.
    assert router._pick_arbiter("a", {"a", "b"}) == "neutral"


def test_pick_arbiter_returns_none_when_no_neutral_alt():
    """If the configured arbiter is in the candidate set AND there's no
    other available model, refuse to escalate/refine — better to skip
    than to ask a candidate to judge itself."""
    config = Config(models={"a": ModelEntry(tags=["math"])})
    router = FleetRouter(config)
    router._registry._available = {"a"}
    router._registry._refreshed = True

    assert router._pick_arbiter("a", {"a"}) is None


@pytest.mark.asyncio
async def test_escalation_swaps_away_from_self_judging_candidate():
    """End-to-end: when the configured escalator IS one of the candidates,
    the actual call must go to a NON-candidate model (or be skipped)."""
    config = Config(
        models={
            "alpha": ModelEntry(tags=["math"], priority=1),
            "beta": ModelEntry(tags=["math"], priority=2),
            "gamma": ModelEntry(tags=["math"], priority=3),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        # Escalator "alpha" will also be dispatched as a candidate (max_parallel=3).
        escalation=EscalationConfig(enabled=True, model="alpha", score_threshold=0.6),
    )
    router = FleetRouter(config)
    router._registry._available = {"alpha", "beta", "gamma"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"alpha": ["1"], "beta": ["2"], "gamma": ["3"]}
        mock_pick.return_value = VerificationResult(
            winner=None,
            all_scored=[
                Candidate("alpha", 0, "1", score=0.2),
                Candidate("beta", 0, "2", score=0.2),
                Candidate("gamma", 0, "3", score=0.2),
            ],
            abstain=True,
        )
        # No candidate for escalator — there's no fourth model. Pick_arbiter
        # should refuse, escalation skipped, abstention surfaced to user.
        result = await router.ask("p")
    # mock_run never called because escalation refused.
    assert mock_run.await_count == 0
    # Abstention path returned to user.
    assert "uncertain" in result


@pytest.mark.asyncio
async def test_refinement_swaps_away_from_winning_model():
    """When the winning model is also the configured critic, refinement
    must rotate to a different model — not ask the winner to critique
    its own answer."""
    config = Config(
        models={
            "winner-and-critic": ModelEntry(tags=["general"], priority=1),
            "neutral": ModelEntry(tags=["general"], priority=2),
        },
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        refinement=RefinementConfig(
            enabled=True, critique_model="winner-and-critic",
        ),
    )
    router = FleetRouter(config)
    router._registry._available = {"winner-and-critic", "neutral"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"winner-and-critic": ["draft"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("winner-and-critic", 0, "draft", score=0.8),
            all_scored=[],
        )
        mock_run.side_effect = [
            {"neutral": "missing X"},      # critique by neutral, not by winner
            {"neutral": "draft plus X"},   # revise by neutral
        ]
        result = await router.ask("explain")
    assert result == "draft plus X"
    # BOTH dispatcher.run calls must have used "neutral", not "winner-and-critic".
    for call in mock_run.call_args_list:
        assert call[0][1] == ["neutral"], (
            f"refinement called {call[0][1]}, but winner was the critic — "
            "self-critique would bias toward 'looks good'"
        )


# ---------- chain-of-thought leak regression guards (BUG 1) ----------


@pytest.mark.asyncio
async def test_thinking_never_leaks_into_returned_winner():
    """A reasoning-style candidate with a <thinking> block must have its
    chain-of-thought stripped before the winner is returned to the user."""
    config = Config(
        models={"model-a": ModelEntry(tags=["math"], priority=1),
                "model-b": ModelEntry(tags=["math"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi:
        mock_multi.return_value = {
            "model-a": ["<thinking>secret reasoning 6*7</thinking>The answer is 42."],
            "model-b": ["The answer is 42."],
        }
        result = await router.ask("what is 6*7")
    assert "<thinking>" not in result
    assert "secret reasoning" not in result
    assert "42" in result


@pytest.mark.asyncio
async def test_thinking_never_leaks_into_abstention_dump():
    """The calibrated-abstention summary dumps candidate text — it must dump
    the stripped final answers, never the raw <thinking> reasoning."""
    config = Config(
        models={"model-a": ModelEntry(tags=["math"], priority=1),
                "model-b": ModelEntry(tags=["math"], priority=2),
                "model-c": ModelEntry(tags=["math"], priority=3)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b", "model-c"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi:
        mock_multi.return_value = {
            "model-a": ["<thinking>secret-a</thinking>The answer is 10."],
            "model-b": ["<thinking>secret-b</thinking>The answer is 20."],
            "model-c": ["<thinking>secret-c</thinking>The answer is 30."],
        }
        result = await router.ask("a hard problem")
    assert "uncertain" in result  # abstention path
    assert "<thinking>" not in result
    for leak in ("secret-a", "secret-b", "secret-c"):
        assert leak not in result


@pytest.mark.asyncio
async def test_thinking_never_leaks_into_escalation_prompt_or_answer():
    config = Config(
        models={"model-a": ModelEntry(tags=["math"], priority=1),
                "model-b": ModelEntry(tags=["math"], priority=2),
                "model-c": ModelEntry(tags=["math"], priority=3),
                "judge": ModelEntry(tags=["math"], priority=4)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        escalation=EscalationConfig(enabled=True, model="judge", score_threshold=0.6),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b", "model-c", "judge"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run:
        mock_multi.return_value = {
            "model-a": ["<thinking>secret-a</thinking>The answer is 10."],
            "model-b": ["<thinking>secret-b</thinking>The answer is 20."],
            "model-c": ["<thinking>secret-c</thinking>The answer is 30."],
        }
        # The judge itself also emits a <thinking> block — its answer must be
        # stripped on the way out too.
        mock_run.return_value = {"judge": "<thinking>weighing options</thinking>42 is correct"}
        result = await router.ask("a hard problem")

    # Escalation answer returned, stripped.
    assert result == "42 is correct"
    # The arbitration prompt embedded the stripped candidate texts only.
    escalate_prompt = mock_run.call_args[0][0]
    assert "<thinking>" not in escalate_prompt
    for leak in ("secret-a", "secret-b", "secret-c"):
        assert leak not in escalate_prompt


@pytest.mark.asyncio
async def test_thinking_stripped_from_refined_answer():
    config = Config(
        models={"model-a": ModelEntry(tags=["general"], priority=1),
                "critic": ModelEntry(tags=["general"], priority=2)},
        synthesis=SynthesisConfig(mode="verifier"),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        refinement=RefinementConfig(enabled=True, critique_model="critic"),
    )
    router = FleetRouter(config)
    router._registry._available = {"model-a", "critic"}
    router._registry._refreshed = True

    with patch.object(router._classifier, "classify", return_value=("general", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["draft answer"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "draft answer", score=0.8),
            all_scored=[],
        )
        mock_run.side_effect = [
            {"critic": "you forgot X"},
            {"critic": "<thinking>let me rewrite</thinking>final revised answer with X"},
        ]
        result = await router.ask("explain")
    assert result == "final revised answer with X"
    assert "<thinking>" not in result


@pytest.mark.asyncio
async def test_thinking_stripped_from_force_model_path():
    config = Config(models={"forced": ModelEntry(tags=["general"], priority=1)})
    router = FleetRouter(config)
    router._registry._available = {"forced"}
    router._registry._refreshed = True

    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {"forced": "<think>secret</think>just the answer"}
        result = await router.ask("hello", force_model="forced")
    assert result == "just the answer"
    assert "<think>" not in result
