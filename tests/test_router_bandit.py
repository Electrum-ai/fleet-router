"""Bandit integration with FleetRouter — selection + reward feedback."""
from unittest.mock import AsyncMock, patch

import pytest

from fleet.config import (
    BanditConfig,
    Config,
    ModelEntry,
    SamplingConfig,
    SynthesisConfig,
)
from fleet.router import FleetRouter
from fleet.verifiers.base import Candidate, VerificationResult


def _config_with_bandit(state_path=""):
    return Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
            "model-c": ModelEntry(tags=["math"], priority=3),
        },
        synthesis=SynthesisConfig(mode="verifier", abstention_threshold=0.0),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        bandit=BanditConfig(enabled=True, state_path=state_path),
    )


def _ready_router(config):
    router = FleetRouter(config)
    router._registry._available = {"model-a", "model-b", "model-c"}
    router._registry._refreshed = True
    return router


@pytest.mark.asyncio
async def test_bandit_disabled_uses_priority_order():
    """Without bandit, selection is priority-sorted from the registry."""
    config = _config_with_bandit()
    config.bandit.enabled = False
    router = _ready_router(config)
    # No bandit instantiated.
    assert router._bandit is None

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["x"], "model-b": ["y"], "model-c": ["z"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "x", score=0.9),
            all_scored=[Candidate("model-a", 0, "x", score=0.9)],
        )
        await router.ask("p")
    dispatched_models = mock_multi.call_args[0][1]
    # With max_parallel=3, all three models go out, in priority order.
    assert dispatched_models == ["model-a", "model-b", "model-c"]


@pytest.mark.asyncio
async def test_bandit_updates_posteriors_from_verifier_scores():
    """Each scored candidate triggers a bandit update."""
    config = _config_with_bandit()
    router = _ready_router(config)
    assert router._bandit is not None

    pre = router._bandit.posterior_mean("math", "model-a")
    assert pre == 0.5  # uniform prior

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {
            "model-a": ["good"], "model-b": ["bad"], "model-c": ["meh"],
        }
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "good", score=1.0),
            all_scored=[
                Candidate("model-a", 0, "good", score=1.0),
                Candidate("model-b", 0, "bad", score=0.0),
                Candidate("model-c", 0, "meh", score=0.5),
            ],
        )
        await router.ask("p")

    # model-a got reward=1.0 → posterior shifts up
    # model-b got reward=0.0 → posterior shifts down
    assert router._bandit.posterior_mean("math", "model-a") > 0.5
    assert router._bandit.posterior_mean("math", "model-b") < 0.5


@pytest.mark.asyncio
async def test_bandit_persists_state(tmp_path):
    """Bandit state survives across router instantiations when state_path is set."""
    state = tmp_path / "bandit.json"
    config = _config_with_bandit(state_path=str(state))
    router = _ready_router(config)
    # Persistence is debounced — force-flush after the single update so
    # the next router can load it. Production CLI uses atexit; tests need
    # an explicit sync point.
    router._bandit._save_every = 1

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-a": ["x"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-a", 0, "x", score=1.0),
            all_scored=[Candidate("model-a", 0, "x", score=1.0)],
        )
        await router.ask("p")

    # New router, same state path — should load the prior update.
    config2 = _config_with_bandit(state_path=str(state))
    router2 = _ready_router(config2)
    assert router2._bandit.posterior_mean("math", "model-a") > 0.5


@pytest.mark.asyncio
async def test_bandit_no_update_when_no_scored_candidates():
    """Verifier with empty all_scored list (catastrophic failure) does not
    crash the bandit update call."""
    config = _config_with_bandit()
    router = _ready_router(config)

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {}
        mock_pick.return_value = VerificationResult(
            winner=None, all_scored=[], rationale="all failed", abstain=True,
        )
        result = await router.ask("p")
    # Posteriors unchanged, no exception raised.
    assert router._bandit.posterior_mean("math", "model-a") == 0.5
    assert "uncertain" in result or "no answer" in result


@pytest.mark.asyncio
async def test_bandit_explores_full_pool_not_just_top_n():
    """With max_parallel=2 and 3 candidate models, the bandit should be able
    to pick model-c (lowest priority) — proving it sees the full pool."""
    import random
    random.seed(0)
    config = _config_with_bandit()
    config.thresholds.max_parallel = 2
    router = _ready_router(config)
    # Skew the bandit so model-c wins draws.
    for _ in range(50):
        router._bandit.update("math", "model-c", 1.0)
        router._bandit.update("math", "model-a", 0.0)
        router._bandit.update("math", "model-b", 0.0)

    with patch.object(router._classifier, "classify", return_value=("math", 0.4)), \
         patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._verifier_synth, "pick", new_callable=AsyncMock) as mock_pick:
        mock_multi.return_value = {"model-c": ["x"]}
        mock_pick.return_value = VerificationResult(
            winner=Candidate("model-c", 0, "x", score=1.0),
            all_scored=[Candidate("model-c", 0, "x", score=1.0)],
        )
        await router.ask("p")

    dispatched = mock_multi.call_args[0][1]
    # model-c (lowest priority) should be in the selected set thanks to bandit.
    assert "model-c" in dispatched
