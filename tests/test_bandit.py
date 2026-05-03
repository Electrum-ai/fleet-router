import json
import os
import random

from fleet.bandit import ThompsonBandit


def test_select_returns_one_of_provided_models():
    b = ThompsonBandit()
    chosen = b.select("code", ["a", "b", "c"])
    assert chosen in {"a", "b", "c"}


def test_select_empty_returns_none():
    b = ThompsonBandit()
    assert b.select("code", []) is None


def test_update_pushes_posterior_toward_observed_reward():
    b = ThompsonBandit()
    # Pre: prior is uniform Beta(1,1), posterior mean = 0.5
    assert b.posterior_mean("code", "good-model") == 0.5
    # 100 successes
    for _ in range(100):
        b.update("code", "good-model", 1.0)
    assert b.posterior_mean("code", "good-model") > 0.95


def test_rank_orders_models_by_draws():
    """After many updates, the better model should rank first more often."""
    random.seed(42)
    b = ThompsonBandit()
    for _ in range(50):
        b.update("code", "good", 1.0)
    for _ in range(50):
        b.update("code", "bad", 0.0)
    # Run many ranks; "good" should be first the majority of the time.
    first_count = sum(1 for _ in range(100) if b.rank("code", ["good", "bad"])[0] == "good")
    assert first_count > 80


def test_persistence_round_trip(tmp_path):
    state = tmp_path / "bandit.json"
    # save_every=1 disables debounce — every update writes immediately.
    b = ThompsonBandit(state_path=str(state), save_every=1)
    b.update("math", "model-x", 1.0)
    b.update("math", "model-x", 0.0)
    b.update("math", "model-x", 1.0)

    # Reload — state persists.
    b2 = ThompsonBandit(state_path=str(state))
    a, beta = b2._params("math", "model-x")
    assert a == 1.0 + 2  # prior + 2 successes
    assert beta == 1.0 + 1


def test_debounce_skips_intermediate_writes(tmp_path):
    """Default save_every (>1) means intermediate updates aren't on disk
    until either the threshold is hit or flush() is called explicitly."""
    state = tmp_path / "bandit.json"
    b = ThompsonBandit(state_path=str(state), save_every=10)
    for _ in range(5):
        b.update("math", "m", 1.0)
    # 5 < 10 → no save yet.
    assert not state.exists()
    b.flush()
    assert state.exists()


def test_concurrent_updates_no_state_corruption(tmp_path):
    """Hammer update() from many threads at once. After all threads finish
    + flush(), the on-disk state must (a) be valid JSON and (b) have the
    correct cumulative alpha+beta given the inputs. Regression guard for
    the .tmp-rename race the v1 save() had."""
    import threading
    state = tmp_path / "bandit.json"
    b = ThompsonBandit(state_path=str(state), save_every=5)

    n_threads = 20
    updates_per_thread = 50

    def worker():
        for _ in range(updates_per_thread):
            b.update("math", "m", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b.flush()

    # Each update added 1.0 to alpha. Total: n_threads * updates_per_thread.
    a, beta = b._params("math", "m")
    assert a == 1.0 + n_threads * updates_per_thread
    assert beta == 1.0  # no failures

    # On-disk file must be parseable JSON, not a half-written .tmp clobber.
    with open(state) as f:
        loaded = json.load(f)
    assert loaded["math"]["m"][0] == a
    assert loaded["math"]["m"][1] == beta


def test_load_handles_corrupt_state(tmp_path):
    state = tmp_path / "bandit.json"
    state.write_text("not valid json {")
    b = ThompsonBandit(state_path=str(state))
    # Should not raise; state is empty.
    assert b.snapshot() == {}


def test_fractional_reward_splits_alpha_beta():
    b = ThompsonBandit()
    b.update("creative", "m", 0.7)
    a, beta = b._params("creative", "m")
    assert abs(a - 1.7) < 1e-6
    assert abs(beta - 1.3) < 1e-6


def test_reward_clipped_to_zero_one():
    b = ThompsonBandit()
    b.update("c", "m", 1.5)  # over
    b.update("c", "m", -0.5)  # under
    a, beta = b._params("c", "m")
    # First call: a += 1.0, b += 0.0
    # Second call: a += 0.0, b += 1.0
    assert abs(a - 2.0) < 1e-6
    assert abs(beta - 2.0) < 1e-6
