"""Fit per-tag abstention thresholds from observed eval outcomes.

This module is the *measured* half of "calibrated abstention": it ties a
verifier's winner_score to observed correctness and chooses, per tag, the
cut-point that delivers a target selective accuracy at the most coverage it
can. The output is a config-ready ``{"abstention_thresholds": {tag: value}}``
block that drops straight into ``SynthesisConfig.abstention_thresholds``.

It is PURE and deterministic: it consumes ``CalibrationRecord`` rows
``(tag, winner_score, correct, abstained)`` and returns a result. No router,
no Ollama, no I/O (except the optional ``write_thresholds`` helper). That makes
the objective unit-testable with synthetic distributions.

Objective (documented, single, clean):

    For each tag, over its ANSWERED observations (not abstained, winner_score
    present), evaluate every observed winner_score as a candidate threshold t.
    The selected set is {obs : winner_score >= t}; its
        selective_accuracy = mean(correct over selected)
        coverage           = |selected| / |answered|
    Choose the LOWEST t whose selected set satisfies
        selective_accuracy >= target_selective_accuracy AND coverage >= min_coverage.
    "Lowest t" maximizes coverage subject to the accuracy guarantee (raising t
    only ever drops the weakest-scoring observations). If no t qualifies, OR the
    tag has fewer than ``min_samples`` answered observations, fall back to the
    global ``default_threshold`` and record a note explaining why.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SELECTIVE_ACCURACY = 0.9
DEFAULT_MIN_COVERAGE = 0.0
DEFAULT_MIN_SAMPLES = 10
DEFAULT_FALLBACK_THRESHOLD = 0.4


@dataclass
class CalibrationRecord:
    """One observation: the verifier's chosen winner_score for a case, whether
    that answer was actually correct, and whether the case abstained.

    ``winner_score`` is None when the case abstained (no winner was returned)
    or when the router emitted no synthesis event.

    ``correct`` for an ANSWERED row is a real measurement (the scorer judged the
    returned answer). For an ABSTAINED row, ``correct`` would mean "the
    SUPPRESSED best candidate was actually correct" — but live eval data never
    measures that (the scorer only sees the abstention-dump string, not the
    suppressed candidate). ``correct_known`` flags whether ``correct`` is a real
    measurement for the row: it is True for answered rows and for synthetic test
    data that supplies true suppressed-candidate correctness, and False for live
    abstained rows. Only rows with ``correct_known=True`` feed
    abstention_precision; when no abstained row is measured, precision is
    reported as None ("not measured") rather than a fabricated 1.0. Threshold
    fitting is unaffected either way — abstained rows are excluded from it."""

    tag: str
    winner_score: Optional[float]
    correct: bool
    abstained: bool = False
    correct_known: bool = True


@dataclass
class TagCalibration:
    """Per-tag fit + a risk-coverage summary."""

    tag: str
    threshold: float
    coverage: float
    selective_accuracy: float
    abstention_rate: float
    abstention_precision: Optional[float]
    n_answered: int
    n_abstained: int
    fallback: bool
    note: str = ""

    def abstention_precision_display(self) -> str:
        """Human-readable precision: the number when measured, else an explicit
        'not measured' note so summaries never imply a fabricated 1.0."""
        if self.abstention_precision is None:
            return "n/a (not measured on live eval data)"
        return f"{self.abstention_precision:.2f}"

    def as_dict(self) -> dict:
        return {
            "tag": self.tag,
            "threshold": self.threshold,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "abstention_rate": self.abstention_rate,
            # Machine-readable: None when not measured (never a fabricated value).
            "abstention_precision": self.abstention_precision,
            # Human-readable: explicit "not measured" instead of an implied 1.0.
            "abstention_precision_display": self.abstention_precision_display(),
            "n_answered": self.n_answered,
            "n_abstained": self.n_abstained,
            "fallback": self.fallback,
            "note": self.note,
        }


@dataclass
class CalibrationResult:
    """Full calibration outcome across all tags."""

    per_tag: dict[str, TagCalibration] = field(default_factory=dict)
    target_selective_accuracy: float = DEFAULT_TARGET_SELECTIVE_ACCURACY
    min_coverage: float = DEFAULT_MIN_COVERAGE
    min_samples: int = DEFAULT_MIN_SAMPLES
    default_threshold: float = DEFAULT_FALLBACK_THRESHOLD

    def config_block(self) -> dict:
        """The config-ready override block. Only tags with a NON-fallback
        fitted threshold are emitted — fallback tags should keep using the
        global ``abstention_threshold`` rather than pinning the default into
        the per-tag map (which would mask future global retuning)."""
        return {
            "abstention_thresholds": {
                tag: round(tc.threshold, 4)
                for tag, tc in sorted(self.per_tag.items())
                if not tc.fallback
            }
        }

    def summary(self) -> dict:
        return {
            "target_selective_accuracy": self.target_selective_accuracy,
            "min_coverage": self.min_coverage,
            "min_samples": self.min_samples,
            "default_threshold": self.default_threshold,
            "tags": {t: tc.as_dict() for t, tc in self.per_tag.items()},
            "config_block": self.config_block(),
        }


def _answered(records: list[CalibrationRecord]) -> list[CalibrationRecord]:
    return [
        r for r in records
        if not r.abstained and r.winner_score is not None
    ]


def _fit_one_tag(
    tag: str,
    records: list[CalibrationRecord],
    *,
    target_selective_accuracy: float,
    min_coverage: float,
    min_samples: int,
    default_threshold: float,
) -> TagCalibration:
    answered = _answered(records)
    abstained = [r for r in records if r.abstained]
    n_answered = len(answered)
    n_abstained = len(abstained)
    total = n_answered + n_abstained
    abstention_rate = (n_abstained / total) if total else 0.0

    # abstention_precision = fraction of abstentions where the best candidate
    # was actually WRONG (i.e. abstaining was the right call). It requires the
    # true correctness of each abstained row's SUPPRESSED top candidate. Live
    # eval data can't supply that (the scorer only sees the abstention-dump
    # string), so those rows are flagged correct_known=False and EXCLUDED here.
    # When no abstained row carries a real measurement, precision is None
    # ("not measured") — never a fabricated value. Only the synthetic/test path,
    # which supplies real suppressed-candidate correctness, produces a number.
    measured_abstained = [r for r in abstained if r.correct_known]
    abstention_precision: Optional[float] = None
    if measured_abstained:
        abstention_precision = (
            sum(1 for r in measured_abstained if not r.correct)
            / len(measured_abstained)
        )

    if n_answered < min_samples:
        return TagCalibration(
            tag=tag,
            threshold=default_threshold,
            coverage=1.0 if n_answered else 0.0,
            selective_accuracy=(
                sum(1 for r in answered if r.correct) / n_answered
                if n_answered else 0.0
            ),
            abstention_rate=abstention_rate,
            abstention_precision=abstention_precision,
            n_answered=n_answered,
            n_abstained=n_abstained,
            fallback=True,
            note=(
                f"only {n_answered} answered sample(s) < min_samples={min_samples}; "
                f"falling back to default threshold {default_threshold}"
            ),
        )

    # Candidate thresholds: every observed winner_score, ascending. Evaluating
    # the observed scores (rather than an arbitrary grid) guarantees the chosen
    # cut-point sits exactly at a real score boundary.
    candidates = sorted({float(r.winner_score) for r in answered})  # type: ignore[arg-type]

    best: Optional[tuple[float, float, float]] = None  # (threshold, sel_acc, coverage)
    for t in candidates:
        selected = [r for r in answered if float(r.winner_score) >= t]  # type: ignore[arg-type]
        if not selected:
            continue
        cov = len(selected) / n_answered
        sel_acc = sum(1 for r in selected if r.correct) / len(selected)
        if sel_acc >= target_selective_accuracy and cov >= min_coverage:
            # Ascending sweep ⇒ first qualifier is the LOWEST threshold ⇒ the
            # most coverage we can buy while keeping the accuracy guarantee.
            best = (t, sel_acc, cov)
            break

    if best is None:
        # No cut-point reaches the target — the score does not separate correct
        # from incorrect for this tag. Fall back to the global default and say
        # so; report the default's own coverage/accuracy for transparency.
        selected = [r for r in answered if float(r.winner_score) >= default_threshold]  # type: ignore[arg-type]
        cov = (len(selected) / n_answered) if n_answered else 0.0
        sel_acc = (
            sum(1 for r in selected if r.correct) / len(selected)
            if selected else 0.0
        )
        return TagCalibration(
            tag=tag,
            threshold=default_threshold,
            coverage=cov,
            selective_accuracy=sel_acc,
            abstention_rate=abstention_rate,
            abstention_precision=abstention_precision,
            n_answered=n_answered,
            n_abstained=n_abstained,
            fallback=True,
            note=(
                f"no threshold reached target selective_accuracy="
                f"{target_selective_accuracy}; scores do not separate "
                f"correct/incorrect — falling back to default {default_threshold}"
            ),
        )

    threshold, sel_acc, cov = best
    return TagCalibration(
        tag=tag,
        threshold=threshold,
        coverage=cov,
        selective_accuracy=sel_acc,
        abstention_rate=abstention_rate,
        abstention_precision=abstention_precision,
        n_answered=n_answered,
        n_abstained=n_abstained,
        fallback=False,
        note=(
            f"lowest threshold reaching selective_accuracy>="
            f"{target_selective_accuracy} at coverage {cov:.2f}"
        ),
    )


def calibrate(
    records: list[CalibrationRecord],
    *,
    target_selective_accuracy: float = DEFAULT_TARGET_SELECTIVE_ACCURACY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    default_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
) -> CalibrationResult:
    """Fit per-tag abstention thresholds from outcome records.

    Deterministic and pure. See the module docstring for the objective. Tags
    with too few answered samples, or for which no cut-point reaches the target
    selective accuracy, fall back to ``default_threshold`` with a logged note.
    """
    by_tag: dict[str, list[CalibrationRecord]] = {}
    for r in records:
        by_tag.setdefault(r.tag, []).append(r)

    result = CalibrationResult(
        target_selective_accuracy=target_selective_accuracy,
        min_coverage=min_coverage,
        min_samples=min_samples,
        default_threshold=default_threshold,
    )
    for tag in sorted(by_tag):
        tc = _fit_one_tag(
            tag, by_tag[tag],
            target_selective_accuracy=target_selective_accuracy,
            min_coverage=min_coverage,
            min_samples=min_samples,
            default_threshold=default_threshold,
        )
        if tc.fallback:
            logger.info("calibrate[%s]: %s", tag, tc.note)
        result.per_tag[tag] = tc
    return result


def write_thresholds(result: CalibrationResult, path) -> None:
    """Write the config-ready ``abstention_thresholds`` block as YAML, nested
    under a ``synthesis:`` key so it can be merged into a fleet config. Also
    emits the per-tag risk-coverage summary as a YAML comment block for the
    operator to eyeball before adopting the thresholds."""
    import yaml  # local import — calibrate() itself stays dependency-free

    block = result.config_block()
    lines = ["# Per-tag abstention thresholds fitted by evals.calibrate.",
             "# Risk-coverage summary (answered/abstained are observation counts):"]
    for tag, tc in sorted(result.per_tag.items()):
        prec = tc.abstention_precision_display()
        lines.append(
            f"#   {tag}: threshold={tc.threshold:.3f} coverage={tc.coverage:.2f} "
            f"selective_accuracy={tc.selective_accuracy:.2f} "
            f"abstention_rate={tc.abstention_rate:.2f} "
            f"abstention_precision={prec} "
            f"n_answered={tc.n_answered}{' (FALLBACK)' if tc.fallback else ''}"
        )
    header = "\n".join(lines) + "\n"
    body = yaml.safe_dump({"synthesis": block}, sort_keys=True)
    from pathlib import Path

    Path(path).write_text(header + body)
