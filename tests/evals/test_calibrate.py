"""Tests for evals.calibrate — pure, deterministic, synthetic data only.

No router, no Ollama: every record is constructed by hand so the objective is
verified directly against (score, correct) distributions.
"""
from pathlib import Path

from evals.calibrate import (
    CalibrationRecord,
    calibrate,
    write_thresholds,
)
from fleet.config import load_config


def _records(tag, scores_correct, abstained=()):
    """Build answered records from [(score, correct), ...] plus optional
    abstained records from [(correct,), ...]."""
    recs = [
        CalibrationRecord(tag=tag, winner_score=s, correct=c, abstained=False)
        for s, c in scores_correct
    ]
    for (c,) in abstained:
        recs.append(
            CalibrationRecord(tag=tag, winner_score=None, correct=c, abstained=True)
        )
    return recs


def test_separable_distribution_yields_obvious_threshold():
    """A clean gap between the wrong cluster (<=0.3) and the correct cluster
    (>=0.85): the fitted threshold is the lowest correct-cluster score, with
    selective_accuracy 1.0 at 0.5 coverage."""
    correct = [(0.85, True), (0.88, True), (0.90, True), (0.92, True), (0.95, True)]
    wrong = [(0.10, False), (0.15, False), (0.20, False), (0.25, False), (0.30, False)]
    result = calibrate(_records("math", correct + wrong))
    tc = result.per_tag["math"]
    assert tc.fallback is False
    assert tc.threshold == 0.85
    assert tc.selective_accuracy == 1.0
    assert tc.coverage == 0.5


def test_inseparable_distribution_falls_back_and_covers_everything():
    """Scores that don't separate correct from incorrect (all at 0.5, half
    right) can't reach the 0.9 target → fall back to the default 0.4, which
    here covers every observation."""
    mixed = [(0.5, i % 2 == 0) for i in range(10)]
    result = calibrate(_records("math", mixed))
    tc = result.per_tag["math"]
    assert tc.fallback is True
    assert tc.threshold == 0.4
    # 0.5 >= 0.4 for all → full coverage on the fallback cut-point.
    assert tc.coverage == 1.0
    assert "no threshold reached target" in tc.note


def test_min_sample_guard_falls_back():
    """A tag with fewer answered samples than min_samples falls back to the
    default with an explanatory note instead of overfitting."""
    few = [(0.9, True), (0.2, False), (0.85, True)]
    result = calibrate(_records("code", few), min_samples=10)
    tc = result.per_tag["code"]
    assert tc.fallback is True
    assert tc.threshold == 0.4
    assert "min_samples" in tc.note


def test_abstention_precision_reported():
    """abstention_precision = fraction of abstentions whose best candidate was
    actually WRONG (abstaining was the right call)."""
    answered = [(0.9, True)] * 6 + [(0.2, False)] * 6
    # 3 abstentions: 2 where the best candidate was wrong, 1 where it was right.
    abstained = [(False,), (False,), (True,)]
    result = calibrate(_records("math", answered, abstained))
    tc = result.per_tag["math"]
    assert tc.n_abstained == 3
    assert tc.abstention_precision == 2 / 3
    # abstention_rate over all observations: 3 / (12 + 3).
    assert tc.abstention_rate == 3 / 15


def test_abstention_precision_not_measured_on_live_data():
    """FIX 3: abstained rows whose suppressed-candidate correctness is UNKNOWN
    (correct_known=False, as the live runner adapter emits) must NOT contribute
    a fabricated precision. With no measured abstained row, precision is None
    and renders as an explicit 'not measured' note — never an implied 1.0."""
    answered = [(0.9, True)] * 6 + [(0.2, False)] * 6
    recs = [
        CalibrationRecord(tag="math", winner_score=s, correct=c, abstained=False)
        for s, c in answered
    ]
    # Three live abstentions: correctness unknown (score was of the dump string).
    for _ in range(3):
        recs.append(CalibrationRecord(
            tag="math", winner_score=None, correct=True,
            abstained=True, correct_known=False,
        ))
    tc = calibrate(recs).per_tag["math"]
    assert tc.n_abstained == 3
    assert tc.abstention_precision is None
    assert tc.abstention_precision_display() == "n/a (not measured on live eval data)"
    # Threshold fitting is unaffected — abstained rows were already excluded.
    assert tc.threshold == 0.9
    assert tc.fallback is False


def test_abstention_precision_partial_measurement_uses_only_known_rows():
    """Mixed: only rows with correct_known=True feed precision; unknown live
    rows are excluded from the numerator AND denominator."""
    answered = [(0.9, True)] * 6 + [(0.2, False)] * 6
    recs = [
        CalibrationRecord(tag="math", winner_score=s, correct=c, abstained=False)
        for s, c in answered
    ]
    # 2 measured abstentions (1 was actually wrong → correct call), 5 unknown.
    recs.append(CalibrationRecord(tag="math", winner_score=None, correct=False, abstained=True))
    recs.append(CalibrationRecord(tag="math", winner_score=None, correct=True, abstained=True))
    for _ in range(5):
        recs.append(CalibrationRecord(
            tag="math", winner_score=None, correct=True,
            abstained=True, correct_known=False,
        ))
    tc = calibrate(recs).per_tag["math"]
    assert tc.n_abstained == 7
    # Precision computed over the 2 measured rows only: 1 wrong / 2 = 0.5.
    assert tc.abstention_precision == 0.5


def test_config_block_excludes_fallback_tags():
    """Only fitted (non-fallback) tags belong in the override block; fallback
    tags should keep using the global default."""
    sep = [(0.85, True), (0.88, True), (0.90, True), (0.92, True), (0.95, True),
           (0.10, False), (0.15, False), (0.20, False), (0.25, False), (0.30, False)]
    few = [(0.9, True), (0.2, False)]
    result = calibrate(_records("math", sep) + _records("code", few))
    block = result.config_block()
    assert "math" in block["abstention_thresholds"]
    assert "code" not in block["abstention_thresholds"]  # fallback excluded


def test_emitted_block_round_trips_into_config(tmp_path):
    """The written thresholds load back into SynthesisConfig.abstention_thresholds
    unchanged — closing the loop from fit → config."""
    sep = [(0.85, True), (0.88, True), (0.90, True), (0.92, True), (0.95, True),
           (0.10, False), (0.15, False), (0.20, False), (0.25, False), (0.30, False)]
    result = calibrate(_records("math", sep))
    out = tmp_path / "thresholds.yaml"
    write_thresholds(result, out)
    cfg = load_config(out)
    assert cfg.synthesis.abstention_thresholds == {"math": 0.85}


def test_calibrate_is_deterministic():
    sep = [(0.85, True), (0.88, True), (0.90, True), (0.92, True), (0.95, True),
           (0.10, False), (0.15, False), (0.20, False), (0.25, False), (0.30, False)]
    r1 = calibrate(_records("math", sep))
    r2 = calibrate(_records("math", sep))
    assert r1.config_block() == r2.config_block()
