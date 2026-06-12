"""Eval harness — runs fixture cases through a router and scores them.

Three output forms:
- Per-case results (for debugging individual failures)
- Per-tag aggregates (mean score, count, pass rate) compared against baseline
- Per-case score lists (one entry per repeat) feeding the paired regression
  gate in ``evals.stats`` — this is what makes the comparison statistically
  sound instead of a raw percentage-point threshold.

Backward compatibility: every previously-public function keeps its old
signature and behaviour. ``repeats`` and per-case data are opt-in extensions.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evals.scorers import (
    CodeExecScorer,
    EvalCase,
    EvalResult,
    KeywordContainsScorer,
    MultipleChoiceScorer,
    NumericMatchScorer,
    Scorer,
)
from evals.stats import (
    DEFAULT_ALPHA,
    DEFAULT_GATE_CONFIDENCE,
    DEFAULT_RESAMPLES,
    PairedVerdict,
    SeedLike,
    paired_regression_verdict,
)

logger = logging.getLogger(__name__)

# Baseline file format version. v1 baselines are flat {tag: aggregate}; v2
# additionally carry per-case results under the reserved "_cases" key so a
# later run can do a paired comparison. Tags never start with an underscore
# (they are words like "math"/"code"), so these reserved keys never collide.
BASELINE_SCHEMA_VERSION = 2
_RESERVED_BASELINE_KEYS = ("_schema", "_cases")
# Deterministic default so the gate's bootstrap CI is reproducible run-to-run.
DEFAULT_GATE_SEED = 0xF1EE7


def default_scorers() -> dict[str, Scorer]:
    """Tag-default and explicit-name scorers. EvalCase.scorer overrides the
    tag default — useful when one tag (e.g. reasoning) needs multiple
    scoring methods across cases."""
    return {
        # Tag defaults
        "code": CodeExecScorer(),
        "math": NumericMatchScorer(),
        "reasoning": KeywordContainsScorer(),
        "summarize": KeywordContainsScorer(),
        "creative": KeywordContainsScorer(),
        "translate": KeywordContainsScorer(),
        "general": KeywordContainsScorer(),
        # Explicit-override names (set EvalCase.scorer to one of these)
        "multi_choice": MultipleChoiceScorer(),
        "code_exec": CodeExecScorer(),
        "numeric": NumericMatchScorer(),
        "keyword": KeywordContainsScorer(),
    }


def case_id(tag: str, prompt: str) -> str:
    """Stable short id for a case, used to key per-case baseline entries.

    A content hash of (tag, prompt) so the same case lines up across runs even
    if fixtures are reordered. NUL-separated so ("a", "bc") != ("ab", "c").
    """
    digest = hashlib.sha256(f"{tag}\x00{prompt}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class PerCaseResult:
    """One case run ``repeats`` times: the raw scores plus derived summaries.

    ``mean_score`` feeds the bootstrap delta; ``passed`` (mean >= 0.5) feeds
    McNemar's pass/fail projection.

    ``winner_scores`` and ``abstained`` are per-repeat (parallel to ``scores``)
    and are populated from the router's ``ResponseSynthesized`` events when the
    router exposes an event bus. They carry the calibration signal: the
    verifier-chosen winner score and whether the case abstained, so the eval
    harness can separate genuine answers from abstention dumps instead of
    scoring the "(uncertain — …)" string as if it were an answer. A repeat with
    no event (non-fleet stub router, or the heuristic fast path) records
    ``winner_score=None`` / ``abstained=False`` — graceful degradation with no
    calibration data rather than a crash.
    """

    case: EvalCase
    id: str
    tag: str
    prompt: str
    scores: list[float] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    notes: str = ""
    winner_scores: list[Optional[float]] = field(default_factory=list)
    abstained: list[bool] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def answered_scores(self) -> list[float]:
        """Scores for repeats that did NOT abstain — what selective accuracy is
        measured over. Falls back to all scores when no abstention data exists
        (e.g. a stub router with no event bus), preserving legacy behavior."""
        if not self.abstained:
            return list(self.scores)
        return [
            s for s, ab in zip(self.scores, self.abstained) if not ab
        ]

    @property
    def pass_fraction(self) -> float:
        """Fraction of repeats that individually passed (score >= 0.5)."""
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 0.5) / len(self.scores)

    @property
    def passed(self) -> bool:
        """Case-level pass: the mean across repeats clears 0.5."""
        return self.mean_score >= 0.5

    def to_baseline_entry(self) -> dict:
        return {
            "tag": self.tag,
            "prompt": self.prompt[:200],
            "scores": list(self.scores),
            "mean_score": self.mean_score,
            "pass_fraction": self.pass_fraction,
            "n_repeats": len(self.scores),
        }


def load_fixtures(directory: Path | str) -> list[EvalCase]:
    """Load all *.jsonl files under `directory`. Each line = one case."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"fixtures directory not found: {directory}")
    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: %s — skipping", path, i, exc)
                    continue
                cases.append(EvalCase(
                    prompt=str(raw.get("prompt", "")),
                    tag=str(raw.get("tag", "general")),
                    expected=raw.get("expected"),
                    test_code=str(raw.get("test_code", "")),
                    scorer=str(raw.get("scorer", "")),
                    metadata=raw.get("metadata", {}) or {},
                ))
    return cases


async def _answer_to_str(router, prompt: str) -> str:
    answer = await router.ask(prompt)
    if isinstance(answer, dict):
        return "\n\n".join(f"--- {k} ---\n{v}" for k, v in answer.items())
    return str(answer)


class _SynthesisCollector:
    """Captures the LAST ResponseSynthesized event emitted on a router's event
    bus during an ask(), so each eval case can be paired with the verifier's
    winner_score and abstain flag. Reset() before each ask() clears stale
    carry-over, so a case that emits no synthesis event records (None, False)
    rather than the previous case's outcome.

    The router emits its synthesis event ONCE, reflecting the FINAL returned
    outcome — i.e. AFTER any escalation/refinement post-pass (see
    ``FleetRouter._parallel``). That makes the last event the collector sees
    match what ask() actually returns: an escalation-rescued case (verifier
    abstained, then a stronger model produced a verified answer) is recorded as
    ANSWERED with the escalated score, not as an abstention. This pairing is
    therefore reliable end-to-end; calibration no longer overcounts abstention
    on the tags where escalation does the most work.

    Subscribes only when the router exposes an EventBus-like ``_events``
    attribute; otherwise stays inert and the harness degrades to no
    calibration data (legacy behavior)."""

    def __init__(self, router) -> None:
        self._bus = getattr(router, "_events", None)
        self.winner_score: Optional[float] = None
        self.abstained: bool = False
        self.tag: Optional[str] = None
        self._active = bool(
            self._bus is not None
            and hasattr(self._bus, "subscribe")
            and hasattr(self._bus, "unsubscribe")
        )

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> "_SynthesisCollector":
        if self._active:
            self._bus.subscribe(self._sink)
        return self

    def __exit__(self, *exc) -> None:
        if self._active:
            self._bus.unsubscribe(self._sink)

    def reset(self) -> None:
        self.winner_score = None
        self.abstained = False
        self.tag = None

    def _sink(self, event) -> None:
        # Imported lazily so evals.runner doesn't hard-depend on fleet internals
        # for the stub-router code paths exercised in unit tests.
        from fleet.events import ResponseSynthesized

        if isinstance(event, ResponseSynthesized):
            self.winner_score = event.winner_score
            self.abstained = bool(event.abstain)
            self.tag = event.tag or None


def _resolve_scorer(case: EvalCase, scorers: dict[str, Scorer]) -> Optional[Scorer]:
    scorer = scorers.get(case.scorer) if case.scorer else None
    if scorer is None:
        scorer = scorers.get(case.tag)
    return scorer


async def run_eval_detailed(
    router,
    cases: list[EvalCase],
    scorers: Optional[dict[str, Scorer]] = None,
    *,
    repeats: int = 1,
) -> list[PerCaseResult]:
    """Run every case ``repeats`` times, returning rich per-case records.

    This is the structure the paired regression gate consumes. ``run_eval``
    is a thin flattening wrapper over this for backward-compatible callers.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    scorers = scorers or default_scorers()
    out: list[PerCaseResult] = []
    with _SynthesisCollector(router) as collector:
        for case in cases:
            scorer = _resolve_scorer(case, scorers)
            if scorer is None:
                logger.warning("no scorer for tag=%s scorer=%s; skipping case",
                               case.tag, case.scorer)
                continue
            scores: list[float] = []
            answers: list[str] = []
            winner_scores: list[Optional[float]] = []
            abstained: list[bool] = []
            notes = ""
            for _ in range(repeats):
                collector.reset()
                answer = await _answer_to_str(router, case.prompt)
                result = await scorer.score(case, answer)
                scores.append(result.score)
                answers.append(result.answer)
                # Pair the scorer outcome with the verifier's synthesis event.
                # An abstained repeat is recorded AS an abstention so downstream
                # selective-accuracy/calibration never treats the abstention
                # dump string as a graded answer.
                winner_scores.append(
                    collector.winner_score if collector.active else None
                )
                abstained.append(collector.abstained if collector.active else False)
                notes = result.notes
            out.append(PerCaseResult(
                case=case,
                id=case_id(case.tag, case.prompt),
                tag=case.tag,
                prompt=case.prompt,
                scores=scores,
                answers=answers,
                notes=notes,
                winner_scores=winner_scores,
                abstained=abstained,
            ))
    return out


async def run_eval(
    router,
    cases: list[EvalCase],
    scorers: Optional[dict[str, Scorer]] = None,
    *,
    repeats: int = 1,
) -> list[EvalResult]:
    """Run every case through `router`, score with the per-tag scorer.

    Returns one EvalResult per case, in input order. With ``repeats=1``
    (the default) behaviour is identical to before. With ``repeats>1`` the
    returned EvalResult carries the mean score across repeats and the last
    repeat's answer/notes; use ``run_eval_detailed`` to get every repeat.
    """
    per_case = await run_eval_detailed(router, cases, scorers, repeats=repeats)
    return [
        EvalResult(
            case=pc.case,
            answer=pc.answers[-1] if pc.answers else "",
            score=pc.mean_score,
            notes=pc.notes,
        )
        for pc in per_case
    ]


def aggregate(results: list[EvalResult]) -> dict[str, dict]:
    """Aggregate by tag: mean score, count, pass rate."""
    by_tag: dict[str, list[EvalResult]] = {}
    for r in results:
        by_tag.setdefault(r.case.tag, []).append(r)
    out: dict[str, dict] = {}
    for tag, rs in by_tag.items():
        scores = [r.score for r in rs]
        passes = sum(1 for s in scores if s >= 0.5)
        out[tag] = {
            "n": len(rs),
            "mean_score": sum(scores) / len(scores),
            "pass_rate": passes / len(rs),
        }
    return out


def aggregate_per_case(per_case: list[PerCaseResult]) -> dict[str, dict]:
    """Aggregate per-case records by tag.

    Legacy keys (``n``, ``mean_score``, ``pass_rate``) are UNCHANGED — they are
    case-level (each case's mean across repeats) and feed the paired regression
    gate exactly as before.

    Calibration keys are added alongside and are computed at the OBSERVATION
    (per-repeat) level, separating each repeat into answered vs abstained:

    - ``coverage``           = answered observations / total observations
    - ``selective_accuracy`` = pass-rate (score >= 0.5) among ANSWERED only
    - ``abstention_rate``    = abstained observations / total observations
    - ``answered``/``abstained`` = the raw observation counts

    When a tag carries no abstention data (stub router with no event bus), every
    observation counts as answered: coverage 1.0, abstention_rate 0.0, and
    selective_accuracy is just the per-observation pass rate — so legacy runs
    are unaffected.
    """
    by_tag: dict[str, list[PerCaseResult]] = {}
    for pc in per_case:
        by_tag.setdefault(pc.tag, []).append(pc)
    out: dict[str, dict] = {}
    for tag, pcs in by_tag.items():
        means = [pc.mean_score for pc in pcs]
        passes = sum(1 for pc in pcs if pc.passed)

        total_obs = 0
        abstained_obs = 0
        answered_correct = 0
        answered_obs = 0
        for pc in pcs:
            for i, s in enumerate(pc.scores):
                total_obs += 1
                is_abstain = pc.abstained[i] if i < len(pc.abstained) else False
                if is_abstain:
                    abstained_obs += 1
                else:
                    answered_obs += 1
                    if s >= 0.5:
                        answered_correct += 1
        out[tag] = {
            "n": len(pcs),
            "mean_score": sum(means) / len(means),
            "pass_rate": passes / len(pcs),
            # calibration (observation-level)
            "answered": answered_obs,
            "abstained": abstained_obs,
            "coverage": (answered_obs / total_obs) if total_obs else 0.0,
            "selective_accuracy": (
                answered_correct / answered_obs if answered_obs else 0.0
            ),
            "abstention_rate": (
                abstained_obs / total_obs if total_obs else 0.0
            ),
        }
    return out


def calibration_records(per_case: list[PerCaseResult]):
    """Flatten per-case records into one ``CalibrationRecord`` per repeat for
    ``evals.calibrate``. Each repeat is an independent observation: its
    verifier winner_score, whether the scored answer was correct (score >=
    0.5), and whether it abstained. Abstained repeats carry winner_score=None
    and are excluded from threshold fitting (recorded as abstentions).

    For an ABSTAINED repeat the per-repeat score ``s`` is the scorer's grade of
    the abstention-DUMP string, NOT of the suppressed best candidate — so it is
    not a real measurement of "was abstaining correct?". Such rows are flagged
    ``correct_known=False`` so ``calibrate`` reports abstention_precision as
    "not measured" rather than collapsing it to a fabricated ~1.0. Answered rows
    carry a real measurement (``correct_known=True``)."""
    from evals.calibrate import CalibrationRecord

    recs = []
    for pc in per_case:
        for i, s in enumerate(pc.scores):
            ws = pc.winner_scores[i] if i < len(pc.winner_scores) else None
            ab = pc.abstained[i] if i < len(pc.abstained) else False
            recs.append(CalibrationRecord(
                tag=pc.tag,
                winner_score=ws,
                # Abstained rows: the score is of the dump string, not the
                # suppressed candidate → correctness is unknown for precision.
                correct=(s >= 0.5),
                abstained=ab,
                correct_known=not ab,
            ))
    return recs


def save_baseline(
    aggregates: dict[str, dict],
    path: Path | str,
    *,
    per_case: Optional[list[PerCaseResult]] = None,
) -> None:
    """Persist a baseline.

    Legacy mode (``per_case=None``) writes exactly the flat {tag: aggregate}
    map, byte-compatible with v1 baselines. When ``per_case`` is supplied the
    file additionally carries per-case score lists under "_cases" and a
    "_schema" version marker, enabling a paired comparison on the next run.
    The flat per-tag keys are kept alongside for readability and back-compat.
    """
    # Reserved-key safety: structural keys are namespaced with a leading
    # underscore ("_schema", "_cases"). A tag that also starts with "_" would
    # silently collide with or overwrite that reserved block, so reject it.
    bad = sorted(t for t in aggregates if isinstance(t, str) and t.startswith("_"))
    if bad:
        raise ValueError(
            f"tag names may not start with '_' (reserved for structural keys "
            f"{_RESERVED_BASELINE_KEYS}); offending tags: {bad}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = dict(aggregates)
    if per_case is not None:
        payload["_schema"] = BASELINE_SCHEMA_VERSION
        payload["_cases"] = {pc.id: pc.to_baseline_entry() for pc in per_case}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _legacy_tags(baseline: dict) -> dict[str, dict]:
    """The flat per-tag aggregates, excluding reserved structural keys."""
    return {k: v for k, v in baseline.items() if k not in _RESERVED_BASELINE_KEYS}


def _paired_verdicts(
    baseline: dict,
    current_per_case: list[PerCaseResult],
    *,
    alpha: float,
    n_resamples: int,
    confidence: float,
    seed: SeedLike,
) ->dict[str, PairedVerdict]:
    """Paired regression verdicts keyed by tag plus an "__overall__" entry.

    Aligns current cases to baseline cases by stable case id, so reordering or
    added/removed fixtures don't corrupt the pairing. Cases missing from either
    side are skipped (and noted by the caller).
    """
    base_cases: dict[str, dict] = baseline.get("_cases", {})
    # Collect aligned (baseline_score, current_score) per tag and overall.
    per_tag: dict[str, tuple[list[float], list[float]]] = {}
    overall_b: list[float] = []
    overall_c: list[float] = []
    for pc in current_per_case:
        entry = base_cases.get(pc.id)
        if entry is None:
            continue
        b_score = float(entry.get("mean_score", 0.0))
        c_score = pc.mean_score
        bs, cs = per_tag.setdefault(pc.tag, ([], []))
        bs.append(b_score)
        cs.append(c_score)
        overall_b.append(b_score)
        overall_c.append(c_score)

    verdicts: dict[str, PairedVerdict] = {}
    for tag, (bs, cs) in per_tag.items():
        verdicts[tag] = paired_regression_verdict(
            bs, cs, alpha=alpha, n_resamples=n_resamples,
            confidence=confidence, seed=seed,
        )
    if overall_b:
        verdicts["__overall__"] = paired_regression_verdict(
            overall_b, overall_c, alpha=alpha, n_resamples=n_resamples,
            confidence=confidence, seed=seed,
        )
    return verdicts


def compare_to_baseline(
    current: dict[str, dict],
    baseline_path: Path | str,
    regression_pp: float = 3.0,
    *,
    current_per_case: Optional[list[PerCaseResult]] = None,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_GATE_CONFIDENCE,
    seed: SeedLike = DEFAULT_GATE_SEED,
) -> tuple[bool, list[str]]:
    """Return (regressed, messages).

    Paired path: when the baseline carries per-case data ("_cases") AND
    ``current_per_case`` is provided, gate on the statistically sound paired
    test from ``evals.stats`` (per tag and overall). A regression is flagged
    only when the current is *worse with significance* — the bootstrap CI on
    the mean delta lies entirely below zero — not on a raw pp threshold.

    Calibration: the gate defaults to ``confidence=DEFAULT_GATE_CONFIDENCE``
    (0.975), which holds the one-sided false-positive rate at or below alpha at
    the small per-tag n this gate targets. The percentile bootstrap is mildly
    anti-conservative at n < ~10, which the higher confidence compensates for.
    Tags with fewer than ``MIN_PAIRED_N`` (3) paired cases abstain rather than
    gate; per-tag gating below ~5 cases is low-power and should be read as such.

    Legacy path: when either side lacks per-case data, fall back to the old
    ``regression_pp`` percentage-point behaviour and log a warning that the
    comparison predates paired gating.
    """
    path = Path(baseline_path)
    if not path.exists():
        return False, [f"no baseline at {path}; current saved as new baseline"]
    with open(path) as f:
        baseline = json.load(f)

    has_baseline_cases = bool(baseline.get("_cases"))
    if has_baseline_cases and current_per_case is not None:
        return _compare_paired(
            baseline, current_per_case,
            alpha=alpha, n_resamples=n_resamples,
            confidence=confidence, seed=seed,
        )

    # ---- legacy percentage-point fallback ----
    if not has_baseline_cases:
        logger.warning(
            "baseline %s predates paired gating (no per-case data); falling "
            "back to a raw %.1fpp threshold — noisy with small n. Re-save the "
            "baseline with per-case data to enable significance testing.",
            path, regression_pp,
        )
    elif current_per_case is None:
        logger.warning(
            "current run has no per-case data; falling back to a raw %.1fpp "
            "threshold. Pass current_per_case (run with repeats) for the "
            "paired gate.", regression_pp,
        )
    return _compare_legacy_pp(current, _legacy_tags(baseline), regression_pp)


def _compare_legacy_pp(
    current: dict[str, dict],
    baseline_tags: dict[str, dict],
    regression_pp: float,
) -> tuple[bool, list[str]]:
    regressed = False
    messages: list[str] = []
    for tag, agg in current.items():
        b = baseline_tags.get(tag)
        if b is None:
            messages.append(f"{tag}: new tag (no baseline)")
            continue
        delta_pp = (agg["pass_rate"] - b.get("pass_rate", 0.0)) * 100
        sign = "+" if delta_pp >= 0 else ""
        messages.append(
            f"{tag}: pass_rate {agg['pass_rate']:.0%} ({sign}{delta_pp:.1f}pp) "
            f"mean {agg['mean_score']:.2f} (n={agg['n']})"
        )
        if delta_pp < -regression_pp:
            regressed = True
    return regressed, messages


def _compare_paired(
    baseline: dict,
    current_per_case: list[PerCaseResult],
    *,
    alpha: float,
    n_resamples: int,
    confidence: float,
    seed: SeedLike,
) ->tuple[bool, list[str]]:
    verdicts = _paired_verdicts(
        baseline, current_per_case,
        alpha=alpha, n_resamples=n_resamples,
        confidence=confidence, seed=seed,
    )
    base_cases = baseline.get("_cases", {})
    current_ids = {pc.id for pc in current_per_case}
    paired_ids = current_ids & set(base_cases)
    new_cases = current_ids - set(base_cases)
    dropped_cases = set(base_cases) - current_ids

    messages: list[str] = []
    messages.append(
        f"paired gate: {len(paired_ids)} cases matched baseline, "
        f"{len(new_cases)} new, {len(dropped_cases)} dropped "
        f"(gate fires on CI upper bound < 0 @ {confidence*100:g}% confidence; "
        f"per-tag gating below ~5 cases is low-power, and the gate abstains "
        f"below 3 paired cases)"
    )

    regressed = False
    # Per-tag lines first, then overall.
    for tag in sorted(k for k in verdicts if k != "__overall__"):
        v = verdicts[tag]
        flag = " REGRESSION" if v.regressed else ""
        messages.append(f"{tag}: {v.detail} n={v.n_pairs}{flag}")
        if v.regressed:
            regressed = True
    overall = verdicts.get("__overall__")
    if overall is not None:
        flag = " REGRESSION" if overall.regressed else ""
        messages.append(f"overall: {overall.detail} n={overall.n_pairs}{flag}")
        if overall.regressed:
            regressed = True
    return regressed, messages


async def run_and_report(
    router,
    fixtures_dir: Path | str,
    baseline_path: Optional[Path | str] = None,
    *,
    repeats: int = 1,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_GATE_CONFIDENCE,
    seed: SeedLike = DEFAULT_GATE_SEED,
) -> dict:
    """End-to-end: load → run (×repeats) → aggregate → compare. Returns a report.

    Threads ``repeats`` through and, when the baseline supports it, includes the
    per-case data and paired significance verdicts in the report.
    """
    cases = load_fixtures(fixtures_dir)
    start = time.time()
    per_case = await run_eval_detailed(router, cases, repeats=repeats)
    elapsed = time.time() - start
    results = [
        EvalResult(
            case=pc.case,
            answer=pc.answers[-1] if pc.answers else "",
            score=pc.mean_score,
            notes=pc.notes,
        )
        for pc in per_case
    ]
    aggregates = aggregate(results)
    # Observation-level calibration view (coverage / selective accuracy /
    # abstention rate per tag), sourced from the rich per-case records.
    cal_agg = aggregate_per_case(per_case)
    calibration = {
        tag: {
            k: v[k]
            for k in (
                "answered", "abstained", "coverage",
                "selective_accuracy", "abstention_rate",
            )
        }
        for tag, v in cal_agg.items()
    }
    report: dict = {
        "n_cases": len(cases),
        "repeats": repeats,
        "elapsed_s": elapsed,
        "aggregates": aggregates,
        "calibration": calibration,
        "per_case": [
            {
                "id": pc.id,
                "tag": pc.tag,
                "prompt": pc.prompt[:200],
                "scores": pc.scores,
                "mean_score": pc.mean_score,
                "pass_fraction": pc.pass_fraction,
            }
            for pc in per_case
        ],
        "results": [
            {
                "tag": r.case.tag,
                "prompt": r.case.prompt[:200],
                "score": r.score,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    if baseline_path is not None:
        regressed, messages = compare_to_baseline(
            aggregates, baseline_path,
            current_per_case=per_case,
            alpha=alpha, n_resamples=n_resamples,
            confidence=confidence, seed=seed,
        )
        report["regressed"] = regressed
        report["comparison"] = messages
        # Attach structured verdicts when the paired path applies.
        path = Path(baseline_path)
        if path.exists():
            with open(path) as f:
                baseline = json.load(f)
            if baseline.get("_cases"):
                verdicts = _paired_verdicts(
                    baseline, per_case,
                    alpha=alpha, n_resamples=n_resamples,
                    confidence=confidence, seed=seed,
                )
                report["gating_method"] = "paired"
                report["significance"] = {
                    k: v.as_dict() for k, v in verdicts.items()
                }
            else:
                report["gating_method"] = "legacy_pp"
    return report
