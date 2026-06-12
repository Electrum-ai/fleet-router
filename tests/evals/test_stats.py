"""Tests for the pure statistical core (evals/stats.py).

Two flavours:
- Hand-computed unit tests of McNemar and the bootstrap (deterministic).
- Self-validating property tests of the gate itself: under the null (no real
  change) it must fire at or below alpha; under a clear regression it must
  fire. These are the checks the old 3pp gate could never pass.
"""
import math
import random

import pytest

from evals.stats import (
    BootstrapResult,
    McNemarResult,
    PairedVerdict,
    is_regression_bootstrap,
    is_regression_mcnemar,
    mcnemar_test,
    paired_bootstrap,
    paired_regression_verdict,
)


# --------------------------------------------------------------------------- #
# McNemar — hand-computed small examples
# --------------------------------------------------------------------------- #

def test_mcnemar_no_discordant_pairs_is_tie():
    # Every pair agrees → no information → p=1, tie.
    base = [True, True, False, False]
    cur = [True, True, False, False]
    r = mcnemar_test(base, cur)
    assert r.baseline_only == 0 and r.current_only == 0
    assert r.p_value == 1.0
    assert r.favors == "tie"
    assert r.method == "degenerate"


def test_mcnemar_counts_discordant_directionally():
    # baseline passes / current fails on 3 cases (b=3); the reverse on 1 (c=1).
    base = [True, True, True, False, True, True]
    cur = [False, False, False, True, True, True]
    r = mcnemar_test(base, cur)
    assert r.baseline_only == 3
    assert r.current_only == 1
    assert r.favors == "baseline"


def test_mcnemar_exact_pvalue_hand_computed():
    # b=4, c=0 → n=4 discordant, k=0. Two-sided exact p = 2 * C(4,0)*0.5^4
    # = 2 * 1/16 = 0.125.
    base = [True, True, True, True]
    cur = [False, False, False, False]
    r = mcnemar_test(base, cur)
    assert r.method == "exact-binomial"
    assert r.p_value == pytest.approx(0.125)
    assert r.favors == "baseline"


def test_mcnemar_exact_symmetric_split_is_insignificant():
    # b=c=3 → perfectly balanced discordance → p should be 1.0, tie.
    base = [True, True, True, False, False, False]
    cur = [False, False, False, True, True, True]
    r = mcnemar_test(base, cur)
    assert r.baseline_only == 3 and r.current_only == 3
    assert r.p_value == pytest.approx(1.0)
    assert r.favors == "tie"


def test_mcnemar_large_uses_chi_square():
    # 60 discordant pairs, all favouring baseline → chi-square branch, tiny p.
    base = [True] * 60 + [True] * 5
    cur = [False] * 60 + [True] * 5
    r = mcnemar_test(base, cur)
    assert r.method == "chi-square-cc"
    # (|60-0|-1)^2 / 60 = 59^2/60 ≈ 58.0
    assert r.statistic == pytest.approx((59 ** 2) / 60)
    assert r.p_value < 1e-6
    assert r.favors == "baseline"


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])


def test_is_regression_mcnemar_requires_baseline_favored_and_significant():
    worse = mcnemar_test([True] * 8, [False] * 8)  # b=8,c=0 → p≈0.0078
    assert worse.p_value < 0.05 and worse.favors == "baseline"
    assert is_regression_mcnemar(worse, alpha=0.05) is True

    better = mcnemar_test([False] * 8, [True] * 8)  # current improved
    assert better.favors == "current"
    assert is_regression_mcnemar(better, alpha=0.05) is False


# --------------------------------------------------------------------------- #
# Bootstrap — determinism and hand-checkable behaviour
# --------------------------------------------------------------------------- #

def test_bootstrap_zero_deltas_collapse_to_point():
    # Identical scores → every delta is 0 → CI is exactly [0, 0].
    r = paired_bootstrap([0.0, 0.0, 0.0, 0.0], seed=1)
    assert r.delta == 0.0
    assert r.low == 0.0
    assert r.high == 0.0
    assert is_regression_bootstrap(r) is False


def test_bootstrap_is_deterministic_for_a_seed():
    deltas = [0.1, -0.2, 0.05, -0.3, 0.0, 0.2]
    r1 = paired_bootstrap(deltas, seed=42, n_resamples=500)
    r2 = paired_bootstrap(deltas, seed=42, n_resamples=500)
    assert (r1.low, r1.high, r1.delta) == (r2.low, r2.high, r2.delta)
    # A Random instance with the same seed gives the same stream/result.
    r3 = paired_bootstrap(deltas, seed=random.Random(42), n_resamples=500)
    assert (r3.low, r3.high) == (r1.low, r1.high)


def test_bootstrap_all_negative_ci_below_zero():
    # Every case worse by a clear margin → CI entirely below zero → regression.
    deltas = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    r = paired_bootstrap(deltas, seed=7)
    assert r.delta == -1.0
    assert r.high < 0.0
    assert is_regression_bootstrap(r) is True


def test_bootstrap_ci_brackets_observed_mean():
    deltas = [0.5, -0.4, 0.3, -0.2, 0.1, -0.1]
    r = paired_bootstrap(deltas, seed=3, n_resamples=1000)
    assert r.low <= r.delta <= r.high


def test_bootstrap_empty_is_safe():
    r = paired_bootstrap([], seed=1)
    assert (r.delta, r.low, r.high, r.n) == (0.0, 0.0, 0.0, 0)
    assert is_regression_bootstrap(r) is False


def test_bootstrap_rejects_bad_confidence():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], confidence=1.5)


# --------------------------------------------------------------------------- #
# Combined verdict
# --------------------------------------------------------------------------- #

def test_verdict_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_regression_verdict([1.0], [1.0, 0.0])


def test_verdict_gate_modes_agree_on_clear_regression():
    base = [1.0] * 8
    cur = [0.0] * 8
    for gate in ("bootstrap", "mcnemar", "both"):
        v = paired_regression_verdict(base, cur, seed=1, gate=gate)
        assert v.regressed is True, gate


def test_verdict_unknown_gate_raises():
    with pytest.raises(ValueError):
        paired_regression_verdict([1.0], [1.0], gate="nonsense")


# --------------------------------------------------------------------------- #
# SELF-VALIDATING property tests — the whole point of the rewrite
# --------------------------------------------------------------------------- #

def _bernoulli_scores(rng: random.Random, n: int, p: float) -> list[float]:
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def _null_fire_rate(n_cases, p, base_offset, n_trials, n_resamples):
    """Fraction of trials the gate fires when baseline≡current in distribution."""
    fires = 0
    for trial in range(n_trials):
        rng = random.Random(base_offset + trial)
        base = _bernoulli_scores(rng, n_cases, p=p)
        cur = _bernoulli_scores(rng, n_cases, p=p)  # same p — no real change
        v = paired_regression_verdict(
            base, cur, seed=trial, n_resamples=n_resamples, gate="bootstrap",
        )
        if v.regressed:
            fires += 1
    return fires / n_trials


# (n_cases, p) bands that exercise the gate at the small n it targets, including
# the worst-measured null case (n=6, p=0.5). Multiple SEED BASES per band so the
# assertion can't pass by seed-luck — every base must independently clear it.
_NULL_BANDS = [(6, 0.5), (8, 0.7)]
_SEED_BASES = [1000, 50_000, 90_000]


@pytest.mark.parametrize("n_cases,p", _NULL_BANDS)
def test_alpha_level_no_false_positives_under_null(n_cases, p):
    """Baseline and current drawn i.i.d. from the SAME distribution.

    There is no real change, so the gate must fire on at most ~alpha of
    independent trials. The old 3pp gate, by contrast, fires constantly here
    because one flipped case in n=8 is a 12.5pp swing.

    Honesty note: the percentile bootstrap is mildly anti-conservative at this
    n. At the *descriptive* 0.95 confidence the true null fire rate runs
    ~0.044-0.052 here — at or slightly above alpha=0.05, so a 120-trial check
    against ``<= 0.05`` was passing on seed-luck (SE ~0.019 at 120 trials).
    The GATE defaults to confidence=0.975, which drops the true rate to ~0.02.
    With 2000 trials the SE is ~0.003, so we assert against 0.04 — real
    Monte-Carlo headroom (~6 SE above a 0.02 true rate) — and require EVERY
    seed base to clear it, not just one.
    """
    n_trials = 2000
    n_resamples = 600
    bound = 0.04  # ~6 SE above the measured ~0.02 true rate at conf=0.975
    for base in _SEED_BASES:
        rate = _null_fire_rate(n_cases, p, base, n_trials, n_resamples)
        se = math.sqrt(rate * (1.0 - rate) / n_trials)
        assert rate <= bound, (
            f"null fire rate {rate:.4f} (SE {se:.4f}) > {bound} "
            f"for n={n_cases} p={p} seed_base={base}"
        )


def test_true_positive_clear_regression_always_fires():
    """Every case drops from pass to fail → the gate MUST fire, every seed."""
    base = [1.0] * 8
    cur = [0.0] * 8
    for seed in range(10):
        v = paired_regression_verdict(base, cur, seed=seed, gate="bootstrap")
        assert v.regressed is True
        assert v.ci_high < 0.0


def test_no_change_but_noisy_does_not_fire():
    """Same multiset of scores, just shuffled across cases (same mean).

    The mean delta is ~0 and the CI straddles zero, so the gate stays quiet —
    where a raw pp gate on a single noisy tag could trip.
    """
    base = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    fired = 0
    for seed in range(30):
        rng = random.Random(seed)
        cur = base[:]
        rng.shuffle(cur)  # identical multiset → identical pass-rate, shuffled
        v = paired_regression_verdict(
            base, cur, seed=seed, n_resamples=800, gate="bootstrap",
        )
        if v.regressed:
            fired += 1
    # Shuffling preserves the mean; the gate should essentially never fire.
    assert fired == 0, f"noisy-but-unchanged fired {fired}/30 times"


def test_partial_real_regression_is_detected_with_enough_signal():
    """A consistent drop across many cases is caught even if not unanimous."""
    # 12 of 14 cases lose half a point; 2 unchanged → mean clearly negative.
    base = [1.0] * 14
    cur = [0.5] * 12 + [1.0, 1.0]
    v = paired_regression_verdict(base, cur, seed=11, n_resamples=1500)
    assert v.delta < 0
    assert v.regressed is True


# --------------------------------------------------------------------------- #
# Small-n abstention guard — tiny tags must NOT fire on a single flip
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2])
def test_small_n_consistent_drop_abstains(n):
    """At n=1/2 the bootstrap CI degenerates (n=1 collapses to a point), so a
    sign-consistent drop would otherwise fire on no real uncertainty. The gate
    must ABSTAIN below MIN_PAIRED_N and say so in the detail."""
    base = [1.0] * n
    cur = [0.0] * n  # every case drops pass→fail, perfectly consistent
    v = paired_regression_verdict(base, cur, seed=1, gate="bootstrap")
    assert v.regressed is False, f"n={n} fired but should abstain"
    assert "too small to gate reliably" in v.detail
    assert f"n={n}" in v.detail
    # The abstention holds across the other gate rules too.
    for gate in ("mcnemar", "both"):
        v2 = paired_regression_verdict(base, cur, seed=1, gate=gate)
        assert v2.regressed is False, f"n={n} gate={gate} fired"


def test_small_n_guard_is_configurable():
    """Lowering min_paired_n re-enables firing at small n (opt-in only)."""
    base = [1.0, 1.0]
    cur = [0.0, 0.0]
    abstains = paired_regression_verdict(base, cur, seed=1)            # default 3
    fires = paired_regression_verdict(base, cur, seed=1, min_paired_n=1)
    assert abstains.regressed is False
    assert fires.regressed is True
    assert "too small" not in fires.detail


def test_min_paired_n_boundary_fires_at_three_consistent_drop():
    """At exactly MIN_PAIRED_N (3) a unanimous drop is allowed to gate."""
    v = paired_regression_verdict([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], seed=1)
    assert v.regressed is True
    assert "too small" not in v.detail


def test_is_regression_bootstrap_abstains_below_min_n():
    """The primitive itself guards: a degenerate n=1 'CI' must not be a
    regression at the default min_n."""
    r = paired_bootstrap([-1.0], seed=1)
    assert (r.low, r.high) == (-1.0, -1.0)  # collapsed to a point
    assert is_regression_bootstrap(r) is False           # default min_n=3
    assert is_regression_bootstrap(r, min_n=1) is True   # opt-in tiny-n
