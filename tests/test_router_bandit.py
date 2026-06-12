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


def _config_with_bandit(state_path="", priority_prior_strength=0.0):
    # priority_prior_strength defaults to 0.0 here (uniform Beta(1,1) arms) so
    # the reward-feedback / exploration tests below keep their clean 0.5 cold
    # start. Cold-start priority seeding is covered by its own tests, which
    # opt into a non-zero strength explicitly.
    return Config(
        models={
            "model-a": ModelEntry(tags=["math"], priority=1),
            "model-b": ModelEntry(tags=["math"], priority=2),
            "model-c": ModelEntry(tags=["math"], priority=3),
        },
        synthesis=SynthesisConfig(mode="verifier", abstention_threshold=0.0),
        sampling=SamplingConfig(samples_by_tag={"math": 1, "default": 1}),
        bandit=BanditConfig(
            enabled=True,
            state_path=state_path,
            priority_prior_strength=priority_prior_strength,
        ),
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


def test_bandit_skips_self_pref_unreliable_result():
    """Case (a) end of FIX 1: a self-preference-biased, multi-model judge
    result (scores_reliable=False) must NOT move any posterior — the cross-
    model scores are biased and would poison the bandit."""
    config = _config_with_bandit()
    router = _ready_router(config)

    biased = VerificationResult(
        winner=Candidate("model-a", 0, "mine", score=1.0),
        all_scored=[
            Candidate("model-a", 0, "mine", score=1.0),
            Candidate("model-b", 0, "theirs", score=0.0),
        ],
        rationale="[self-preference: ...] ",
        scores_reliable=False,
    )
    router._update_bandit("math", biased)

    # Untouched: both still at the uniform prior.
    assert router._bandit.posterior_mean("math", "model-a") == 0.5
    assert router._bandit.posterior_mean("math", "model-b") == 0.5


def test_bandit_applies_self_consistency_reliable_result():
    """Case (b) end of FIX 1: a single-model self-consistency judge result is
    reliable (scores_reliable=True) and DOES update the bandit — the
    self-pref note doesn't suppress legitimate single-pool signal."""
    config = _config_with_bandit()
    router = _ready_router(config)

    reliable = VerificationResult(
        winner=Candidate("model-a", 0, "s1", score=1.0),
        all_scored=[
            Candidate("model-a", 0, "s1", score=1.0),
            Candidate("model-a", 1, "s2", score=1.0),
        ],
        rationale="[self-preference: ...] ",
        scores_reliable=True,
    )
    router._update_bandit("math", reliable)

    # Two reliable model-a candidates aggregate to ONE update (mean reward
    # 1.0), so the posterior shifts up — just once, not twice (FIX 1).
    assert router._bandit.posterior_mean("math", "model-a") > 0.5


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


# --- FIX 1: per-round aggregation -------------------------------------------

def _spy_updates(bandit):
    """Wrap bandit.update to record (tag, model, reward) and still apply it."""
    calls = []
    real = bandit.update

    def spy(tag, model, reward):
        calls.append((tag, model, reward))
        real(tag, model, reward)

    bandit.update = spy
    return calls


def test_per_round_aggregation_one_update_per_model():
    """5 correlated candidates from one model → exactly ONE bandit update with
    reward = mean candidate score, NOT five Bernoulli updates."""
    config = _config_with_bandit()  # uniform prior (strength 0.0)
    router = _ready_router(config)
    calls = _spy_updates(router._bandit)

    result = VerificationResult(
        winner=Candidate("model-a", 0, "a", score=1.0),
        all_scored=[
            Candidate("model-a", 0, "a", score=1.0),
            Candidate("model-a", 1, "b", score=1.0),
            Candidate("model-a", 2, "c", score=1.0),
            Candidate("model-a", 3, "d", score=0.0),
            Candidate("model-a", 4, "e", score=0.0),
        ],
    )
    router._update_bandit("math", result)

    assert len(calls) == 1
    assert calls[0] == ("math", "model-a", 0.6)  # mean of [1,1,1,0,0]
    # Posterior reflects a SINGLE 0.6 observation: alpha=1.6, beta=1.4.
    mean = router._bandit.posterior_mean("math", "model-a")
    assert abs(mean - (1.6 / 3.0)) < 1e-6
    # And NOT the overconfident five-update result (alpha=4, beta=3 → 4/7).
    assert abs(mean - (4 / 7)) > 1e-3


def test_per_round_aggregation_groups_by_model():
    """Candidates are grouped per model: one mean-reward update each."""
    config = _config_with_bandit()
    router = _ready_router(config)
    calls = _spy_updates(router._bandit)

    result = VerificationResult(
        winner=Candidate("model-a", 0, "x", score=1.0),
        all_scored=[
            Candidate("model-a", 0, "x", score=1.0),
            Candidate("model-a", 1, "y", score=0.0),   # model-a mean = 0.5
            Candidate("model-b", 0, "z", score=0.5),
            Candidate("model-b", 1, "w", score=0.5),
            Candidate("model-b", 2, "v", score=0.5),    # model-b mean = 0.5
        ],
    )
    router._update_bandit("math", result)

    assert len(calls) == 2
    by_model = {m: r for _, m, r in calls}
    assert abs(by_model["model-a"] - 0.5) < 1e-6
    assert abs(by_model["model-b"] - 0.5) < 1e-6


# --- FIX 3: cold-start priority seeding -------------------------------------

def test_cold_start_seeds_priority_prior():
    """A fresh bandit seeds priority-1 with the highest prior mean."""
    config = _config_with_bandit(priority_prior_strength=0.5)
    router = _ready_router(config)
    means = {
        m: router._bandit.posterior_mean("math", m)
        for m in ("model-a", "model-b", "model-c")
    }
    # priority 1 > 2 > 3 in prior mean.
    assert means["model-a"] > means["model-b"] > means["model-c"]
    # model-a: alpha = 1 + 0.5/1 = 1.5 → mean 1.5/2.5 = 0.6.
    assert abs(means["model-a"] - 0.6) < 1e-6


def test_cold_start_strength_zero_is_uniform():
    """strength=0.0 disables seeding — every arm at the uniform 0.5 prior."""
    config = _config_with_bandit(priority_prior_strength=0.0)
    router = _ready_router(config)
    assert router._bandit._prior_provider is None
    for m in ("model-a", "model-b", "model-c"):
        assert router._bandit.posterior_mean("math", m) == 0.5


def test_cold_start_priority_one_selected_most_often():
    """Over many fresh draws, the priority-1 seed wins rank() most often."""
    import random
    from collections import Counter

    random.seed(0)
    config = _config_with_bandit(priority_prior_strength=2.0)  # wider gap
    router = _ready_router(config)
    pool = ["model-a", "model-b", "model-c"]
    firsts = [router._bandit.rank("math", pool)[0] for _ in range(500)]
    counts = Counter(firsts)
    assert counts["model-a"] == max(counts.values())


def test_evidence_overtakes_priority_seed():
    """A low-priority but high-reward model overtakes the priority-1 seed once
    enough real observations accumulate — evidence dominates the mild prior."""
    config = _config_with_bandit(priority_prior_strength=0.5)
    router = _ready_router(config)
    # model-c starts seeded lowest but turns out best; model-a (seeded high)
    # turns out worst.
    for _ in range(30):
        router._update_bandit("math", VerificationResult(
            winner=Candidate("model-c", 0, "x", score=1.0),
            all_scored=[Candidate("model-c", 0, "x", score=1.0)],
        ))
        router._update_bandit("math", VerificationResult(
            winner=Candidate("model-a", 0, "y", score=0.0),
            all_scored=[Candidate("model-a", 0, "y", score=0.0)],
        ))
    assert (
        router._bandit.posterior_mean("math", "model-c")
        > router._bandit.posterior_mean("math", "model-a")
    )
