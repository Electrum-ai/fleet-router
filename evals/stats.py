"""Statistical core for eval regression gating — PURE and deterministic.

No router, no I/O, no global state. Everything here is a plain function over
lists of numbers, so it is trivially unit-testable and reproducible: every
randomized routine takes an explicit `seed` (or a `random.Random`).

The eval harness historically gated regressions on a raw 3-percentage-point
pass-rate delta. With n~=6-8 cases per tag and temperature-0.7 sampling, a
single case flipping is 12-17pp — so that gate both missed real regressions
and fired on pure noise. This module replaces the point estimate with a
paired comparison that carries an uncertainty interval:

- ``paired_bootstrap`` — resamples per-case score deltas to produce a
  confidence interval on the mean delta. The gate fires only when the whole
  interval is below zero (current confidently worse).
- ``mcnemar_test`` — the classic paired test for binary pass/fail outcomes,
  reported alongside the bootstrap as corroboration.
- ``paired_regression_verdict`` — ties them together into a single decision.

Sign convention everywhere: ``delta = current - baseline``. Negative is worse.

Calibration honesty
-------------------
The percentile bootstrap is *mildly anti-conservative* at the small sample
sizes this gate targets (n < ~10): at the descriptive 95% confidence its
measured one-sided "high < 0" false-positive rate under H0 runs ~0.046-0.063
at n=6-8 — i.e. at or slightly above a nominal alpha=0.05. The regression
GATE therefore defaults to ``DEFAULT_GATE_CONFIDENCE = 0.975`` (a one-sided
upper tail of 1.25%), which the reviewer's Monte-Carlo measured drops the
false-positive rate to ~0.02-0.03 at n=6 p=0.5 and n=8 p=0.7 with no power
loss on real, consistent per-case drops. The higher gate confidence is what
keeps the one-sided false-positive rate at or below alpha; the descriptive
side-by-side interval in ``evals.compare`` stays at the conventional
``DEFAULT_CONFIDENCE = 0.95`` because it reports an interval, it does not gate.

Per-tag gating below ~5 cases is inherently low-power, and at n < 3 the
bootstrap interval degenerates (n=1 collapses to a point). ``MIN_PAIRED_N``
makes the gate abstain there rather than fire on a single sign-consistent
flip.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence, Union

# A seed may be an int, an existing Random (to share a stream), or None.
SeedLike = Union[int, random.Random, None]

DEFAULT_RESAMPLES = 2000
# Descriptive interval confidence (side-by-side reporting in evals.compare).
DEFAULT_CONFIDENCE = 0.95
# Regression-GATE confidence. Higher than the descriptive 0.95 to compensate
# for the percentile bootstrap's mild anti-conservatism at n < ~10: a 0.975
# two-sided interval is a one-sided upper tail of 1.25%, which holds the
# false-positive rate at or below alpha at the small n the gate targets.
DEFAULT_GATE_CONFIDENCE = 0.975
DEFAULT_ALPHA = 0.05
# Below this many paired cases the bootstrap interval is too degenerate to
# gate on (n=1 collapses to a point; n=2-3 fires on any sign-consistent drop).
# The gate abstains — returns "not a regression" — below this threshold.
MIN_PAIRED_N = 3
# Above this many discordant pairs, McNemar switches from the exact binomial
# to the chi-square approximation with continuity correction.
EXACT_MCNEMAR_MAX = 50


def _as_random(seed: SeedLike) -> random.Random:
    """Coerce a seed-like value into a ``random.Random`` instance."""
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    ``q`` is a fraction in [0, 1]. Matches numpy's default ('linear')
    interpolation so results are comparable, but needs no numpy.
    """
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_xs[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_xs[int(lo)]
    frac = pos - lo
    return sorted_xs[int(lo)] * (1.0 - frac) + sorted_xs[int(hi)] * frac


# --------------------------------------------------------------------------- #
# Paired bootstrap over per-case score deltas
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BootstrapResult:
    """Confidence interval on the mean of per-case deltas (current - baseline)."""

    delta: float          # observed mean delta
    low: float            # CI lower bound
    high: float           # CI upper bound
    confidence: float     # e.g. 0.95
    n: int                # number of paired cases
    n_resamples: int


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: SeedLike = None,
) -> BootstrapResult:
    """Percentile bootstrap CI on the mean of ``deltas``.

    Each delta is one paired case's (current_score - baseline_score). We
    resample cases with replacement ``n_resamples`` times, take the mean of
    each resample, and read the ``confidence`` central interval off the
    resampled means. Deterministic given ``seed``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    n = len(deltas)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, confidence, 0, n_resamples)

    observed = sum(deltas) / n
    if n == 1:
        # No resampling variation possible — the interval collapses to the point.
        return BootstrapResult(observed, observed, observed, confidence, 1, n_resamples)

    rng = _as_random(seed)
    deltas = list(deltas)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low = _percentile(means, tail)
    high = _percentile(means, 1.0 - tail)
    return BootstrapResult(observed, low, high, confidence, n, n_resamples)


def is_regression_bootstrap(
    result: BootstrapResult, *, min_n: int = MIN_PAIRED_N
) -> bool:
    """Regression iff the entire CI sits below zero (current confidently worse).

    Abstains (returns ``False``) when ``result.n < min_n``: below ~3 paired
    cases the percentile bootstrap interval degenerates — at n=1 it collapses
    to a point and fires on a single flip — which would reintroduce exactly the
    noise-driven gating this module exists to remove.
    """
    if result.n < min_n:
        return False
    return result.n > 0 and result.high < 0.0


# --------------------------------------------------------------------------- #
# McNemar's test for paired binary pass/fail outcomes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class McNemarResult:
    n_pairs: int
    baseline_only: int    # b: baseline passed, current failed (regressions)
    current_only: int     # c: baseline failed, current passed (improvements)
    statistic: float      # chi-square statistic (nan for the exact branch)
    p_value: float        # two-sided
    favors: str           # "baseline" | "current" | "tie"
    method: str           # "exact-binomial" | "chi-square-cc" | "degenerate"


def _binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar's discordant pairs.

    Under H0 each discordant pair is equally likely to favour either side, so
    the count favouring one side is Binomial(n=b+c, p=0.5). The two-sided
    p-value doubles the smaller tail (capped at 1.0).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof = erfc(sqrt(x/2))."""
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_test(
    baseline_pass: Sequence[bool],
    current_pass: Sequence[bool],
    *,
    exact_max: int = EXACT_MCNEMAR_MAX,
) -> McNemarResult:
    """Paired test on binary pass/fail outcomes, aligned by position.

    Only the discordant pairs (one side passes, the other fails) carry
    information. Uses the exact binomial p-value for small discordant counts
    and the chi-square approximation with continuity correction for large.
    """
    if len(baseline_pass) != len(current_pass):
        raise ValueError(
            f"length mismatch: {len(baseline_pass)} != {len(current_pass)}"
        )
    n_pairs = len(baseline_pass)
    b = c = 0
    for bp, cp in zip(baseline_pass, current_pass):
        if bp and not cp:
            b += 1
        elif cp and not bp:
            c += 1
    n = b + c
    if n == 0:
        return McNemarResult(n_pairs, 0, 0, 0.0, 1.0, "tie", "degenerate")

    if n <= exact_max:
        p = _binom_two_sided_p(b, c)
        stat = float("nan")
        method = "exact-binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p = _chi2_sf_1df(stat)
        method = "chi-square-cc"

    favors = "baseline" if b > c else ("current" if c > b else "tie")
    return McNemarResult(n_pairs, b, c, stat, p, favors, method)


def is_regression_mcnemar(result: McNemarResult, alpha: float = DEFAULT_ALPHA) -> bool:
    """Regression iff significant AND the discordant pairs favour the baseline."""
    return result.favors == "baseline" and result.p_value < alpha


# --------------------------------------------------------------------------- #
# Combined verdict
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairedVerdict:
    """The gate's decision for one comparison (a tag, or the overall set)."""

    n_pairs: int
    delta: float          # observed mean score delta (current - baseline)
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float        # McNemar two-sided
    discordant_baseline: int
    discordant_current: int
    regressed: bool
    gate: str             # which rule decided: "bootstrap" | "mcnemar" | "both"
    detail: str

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "discordant_baseline": self.discordant_baseline,
            "discordant_current": self.discordant_current,
            "regressed": self.regressed,
            "gate": self.gate,
            "detail": self.detail,
        }


def paired_regression_verdict(
    baseline_scores: Sequence[float],
    current_scores: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_GATE_CONFIDENCE,
    seed: SeedLike = None,
    pass_threshold: float = 0.5,
    gate: str = "bootstrap",
    min_paired_n: int = MIN_PAIRED_N,
) -> PairedVerdict:
    """Decide whether ``current`` regressed vs ``baseline`` on paired scores.

    Both inputs are per-case mean scores, aligned by position (case i in
    baseline corresponds to case i in current). Computes:

      - a bootstrap CI on the mean delta (continuous, the default gate), and
      - McNemar's test on the pass/fail projection (corroboration).

    ``confidence`` defaults to ``DEFAULT_GATE_CONFIDENCE`` (0.975), *higher*
    than the descriptive 0.95 interval, because the percentile bootstrap is
    mildly anti-conservative at the small n this gate targets (n < ~10). The
    higher confidence is what holds the one-sided false-positive rate at or
    below ``alpha`` at those sample sizes; see the module docstring for the
    measured calibration.

    ``gate`` selects the decision rule:
      - "bootstrap" (default): fire iff the bootstrap CI is entirely < 0.
        This is one-sided at (1-confidence)/2 (1.25% at the default 0.975).
      - "mcnemar": fire iff McNemar is significant and favours the baseline.
      - "both": fire only when bootstrap AND McNemar agree (most conservative).

    ``min_paired_n`` (default ``MIN_PAIRED_N`` = 3): below this many paired
    cases the gate ABSTAINS — ``regressed`` is forced ``False`` and a
    "n=<k> too small to gate reliably" note is appended to ``detail`` — because
    the bootstrap interval degenerates at tiny n. Per-tag gating below ~5 cases
    is low-power regardless and should be read with that caveat.
    """
    if len(baseline_scores) != len(current_scores):
        raise ValueError(
            f"length mismatch: {len(baseline_scores)} != {len(current_scores)}"
        )
    deltas = [c - b for b, c in zip(baseline_scores, current_scores)]
    boot = paired_bootstrap(
        deltas, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    mc = mcnemar_test(
        [b >= pass_threshold for b in baseline_scores],
        [c >= pass_threshold for c in current_scores],
    )

    too_small = boot.n < min_paired_n
    boot_reg = is_regression_bootstrap(boot, min_n=min_paired_n)
    mc_reg = is_regression_mcnemar(mc, alpha=alpha)
    if gate == "bootstrap":
        regressed = boot_reg
    elif gate == "mcnemar":
        regressed = mc_reg
    elif gate == "both":
        regressed = boot_reg and mc_reg
    else:
        raise ValueError(f"unknown gate {gate!r} (bootstrap|mcnemar|both)")

    # Abstain below the minimum sample size regardless of gate rule: McNemar is
    # equally low-power at n < 3, so the higher-confidence bootstrap guard alone
    # is not enough to keep the "both"/"mcnemar" paths honest.
    if too_small:
        regressed = False

    pct = f"{confidence * 100:g}"
    detail = (
        f"delta={boot.delta:+.3f} "
        f"CI[{boot.low:+.3f},{boot.high:+.3f}]@{pct}% "
        f"McNemar p={mc.p_value:.3f} "
        f"(worse={mc.baseline_only}, better={mc.current_only})"
    )
    if too_small:
        detail += f" [n={boot.n} too small to gate reliably; abstaining]"
    return PairedVerdict(
        n_pairs=boot.n,
        delta=boot.delta,
        ci_low=boot.low,
        ci_high=boot.high,
        confidence=confidence,
        p_value=mc.p_value,
        discordant_baseline=mc.baseline_only,
        discordant_current=mc.current_only,
        regressed=regressed,
        gate=gate,
        detail=detail,
    )
