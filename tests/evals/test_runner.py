"""Tests for the eval harness — uses a stub router so no Ollama needed."""
import json
from pathlib import Path

import pytest

from evals.runner import (
    aggregate,
    aggregate_per_case,
    case_id,
    compare_to_baseline,
    load_fixtures,
    run_eval,
    run_eval_detailed,
    save_baseline,
)
from evals.scorers import EvalCase, EvalResult, KeywordContainsScorer, NumericMatchScorer


class _StubRouter:
    """Returns a fixed answer per prompt prefix."""

    def __init__(self, answers: dict[str, str]):
        self._answers = answers

    async def ask(self, prompt: str):
        for prefix, ans in self._answers.items():
            if prompt.startswith(prefix):
                return ans
        return "no answer"


class _SequenceRouter:
    """Returns the next canned answer per prompt on each call — lets a test
    simulate per-repeat nondeterminism without any real sampling."""

    def __init__(self, sequences: dict[str, list[str]]):
        self._sequences = {k: list(v) for k, v in sequences.items()}
        self._idx: dict[str, int] = {}

    async def ask(self, prompt: str):
        seq = self._sequences.get(prompt, ["no answer"])
        i = self._idx.get(prompt, 0)
        self._idx[prompt] = i + 1
        return seq[min(i, len(seq) - 1)]


def test_load_fixtures_reads_jsonl(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2*3?", "tag": "math", "expected": 6}\n'
    )
    cases = load_fixtures(tmp_path)
    assert len(cases) == 2
    assert cases[0].prompt == "1+1?"
    assert cases[0].expected == 2


def test_load_fixtures_skips_blank_and_comments(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(
        '\n'
        '# this is a comment\n'
        '{"prompt": "p", "tag": "general", "expected": ["x"]}\n'
    )
    assert len(load_fixtures(tmp_path)) == 1


def test_load_fixtures_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fixtures(tmp_path / "nope")


@pytest.mark.asyncio
async def test_run_eval_scores_each_case():
    cases = [
        EvalCase(prompt="math1", tag="math", expected=42),
        EvalCase(prompt="math2", tag="math", expected=10),
    ]
    router = _StubRouter({"math1": "the answer is 42", "math2": "I think 99"})
    scorers = {"math": NumericMatchScorer()}
    results = await run_eval(router, cases, scorers)
    assert len(results) == 2
    assert results[0].score == 1.0
    assert results[1].score == 0.0


@pytest.mark.asyncio
async def test_run_eval_handles_dict_answer():
    cases = [EvalCase(prompt="q", tag="general", expected=["foo"])]
    class DictRouter:
        async def ask(self, p):
            return {"a": "foo content", "b": "other"}
    results = await run_eval(DictRouter(), cases, {"general": KeywordContainsScorer()})
    assert results[0].score == 1.0


def test_aggregate_per_tag():
    results = [
        EvalResult(case=EvalCase(prompt="", tag="math"), answer="", score=1.0),
        EvalResult(case=EvalCase(prompt="", tag="math"), answer="", score=0.0),
        EvalResult(case=EvalCase(prompt="", tag="code"), answer="", score=1.0),
    ]
    agg = aggregate(results)
    assert agg["math"]["n"] == 2
    assert agg["math"]["mean_score"] == 0.5
    assert agg["math"]["pass_rate"] == 0.5
    assert agg["code"]["pass_rate"] == 1.0


def test_compare_to_baseline_no_baseline(tmp_path):
    regressed, msgs = compare_to_baseline({}, tmp_path / "missing.json")
    assert not regressed
    assert "no baseline" in msgs[0]


def test_compare_to_baseline_detects_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    save_baseline({"math": {"n": 5, "mean_score": 0.9, "pass_rate": 0.9}}, baseline)
    current = {"math": {"n": 5, "mean_score": 0.5, "pass_rate": 0.5}}
    regressed, msgs = compare_to_baseline(current, baseline, regression_pp=3.0)
    assert regressed
    assert any("math" in m for m in msgs)


def test_compare_to_baseline_within_tolerance(tmp_path):
    baseline = tmp_path / "b.json"
    save_baseline({"math": {"n": 5, "mean_score": 0.8, "pass_rate": 0.8}}, baseline)
    current = {"math": {"n": 5, "mean_score": 0.78, "pass_rate": 0.79}}
    regressed, _ = compare_to_baseline(current, baseline, regression_pp=3.0)
    assert not regressed


# --------------------------------------------------------------------------- #
# repeats + per-case structure
# --------------------------------------------------------------------------- #

def test_case_id_is_stable_and_distinguishes_fields():
    assert case_id("math", "1+1?") == case_id("math", "1+1?")
    assert case_id("math", "1+1?") != case_id("code", "1+1?")
    # NUL-separated so concatenation collisions can't happen.
    assert case_id("a", "bc") != case_id("ab", "c")


@pytest.mark.asyncio
async def test_run_eval_detailed_records_one_score_per_repeat():
    """repeats=3 with a deterministic stub records 3 scores per case and the
    paired path can consume them — no Ollama involved."""
    cases = [
        EvalCase(prompt="math1", tag="math", expected=42),
        EvalCase(prompt="math2", tag="math", expected=10),
    ]
    # math1 alternates right/wrong/right; math2 always wrong.
    router = _SequenceRouter({
        "math1": ["answer 42", "answer 0", "answer 42"],
        "math2": ["answer 1", "answer 2", "answer 3"],
    })
    scorers = {"math": NumericMatchScorer()}
    per_case = await run_eval_detailed(router, cases, scorers, repeats=3)
    assert len(per_case) == 2
    assert per_case[0].scores == [1.0, 0.0, 1.0]
    assert per_case[0].mean_score == pytest.approx(2 / 3)
    assert per_case[0].pass_fraction == pytest.approx(2 / 3)
    assert per_case[0].passed is True       # mean 0.67 >= 0.5
    assert per_case[1].scores == [0.0, 0.0, 0.0]
    assert per_case[1].passed is False
    assert per_case[0].id == case_id("math", "math1")


@pytest.mark.asyncio
async def test_run_eval_repeats_one_matches_legacy_shape():
    cases = [EvalCase(prompt="math1", tag="math", expected=42)]
    router = _StubRouter({"math1": "the answer is 42"})
    results = await run_eval(router, cases, {"math": NumericMatchScorer()})
    assert len(results) == 1 and results[0].score == 1.0


@pytest.mark.asyncio
async def test_run_eval_repeats_returns_mean_score():
    cases = [EvalCase(prompt="math1", tag="math", expected=42)]
    router = _SequenceRouter({"math1": ["answer 42", "answer 0"]})
    results = await run_eval(router, cases, {"math": NumericMatchScorer()}, repeats=2)
    assert results[0].score == pytest.approx(0.5)  # mean of [1, 0]


def test_run_eval_detailed_rejects_zero_repeats():
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(run_eval_detailed(_StubRouter({}), [], repeats=0))


@pytest.mark.asyncio
async def test_aggregate_per_case_matches_aggregate():
    cases = [
        EvalCase(prompt="math1", tag="math", expected=42),
        EvalCase(prompt="math2", tag="math", expected=10),
    ]
    router = _StubRouter({"math1": "answer 42", "math2": "answer 0"})
    per_case = await run_eval_detailed(router, cases, {"math": NumericMatchScorer()})
    agg = aggregate_per_case(per_case)
    assert agg["math"]["n"] == 2
    assert agg["math"]["pass_rate"] == 0.5


# --------------------------------------------------------------------------- #
# baseline schema + paired gating
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_save_baseline_persists_per_case_block():
    cases = [EvalCase(prompt="math1", tag="math", expected=42)]
    router = _StubRouter({"math1": "answer 42"})
    per_case = await run_eval_detailed(router, cases, {"math": NumericMatchScorer()})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "baseline.json"
        save_baseline(aggregate_per_case(per_case), path, per_case=per_case)
        data = json.loads(path.read_text())
        # Legacy per-tag keys preserved...
        assert data["math"]["pass_rate"] == 1.0
        # ...plus the new schema marker and per-case block.
        assert data["_schema"] == 2
        cid = case_id("math", "math1")
        assert cid in data["_cases"]
        assert data["_cases"][cid]["scores"] == [1.0]


def test_save_baseline_legacy_mode_writes_flat_map(tmp_path):
    """Without per_case the file is byte-compatible with v1 baselines."""
    path = tmp_path / "b.json"
    save_baseline({"math": {"n": 5, "mean_score": 0.9, "pass_rate": 0.9}}, path)
    data = json.loads(path.read_text())
    assert set(data) == {"math"}
    assert "_cases" not in data and "_schema" not in data


def test_save_baseline_rejects_reserved_underscore_tag(tmp_path):
    """A tag starting with '_' would collide with the reserved structural keys
    ('_schema'/'_cases'); save_baseline must reject it rather than corrupt the
    per-case block."""
    path = tmp_path / "b.json"
    with pytest.raises(ValueError, match=r"may not start with '_'"):
        save_baseline({"_cases": {"n": 1, "mean_score": 1.0, "pass_rate": 1.0}}, path)
    assert not path.exists()  # nothing written on rejection


def _build_baseline(tmp_path, prompts_scores):
    """Helper: write a v2 baseline from {prompt: [scores]} under tag 'math'."""
    from evals.runner import PerCaseResult
    per_case = []
    agg_scores = []
    for prompt, scores in prompts_scores.items():
        pc = PerCaseResult(
            case=EvalCase(prompt=prompt, tag="math"),
            id=case_id("math", prompt),
            tag="math",
            prompt=prompt,
            scores=list(scores),
            answers=["x"] * len(scores),
        )
        per_case.append(pc)
        agg_scores.append(pc.mean_score)
    agg = {"math": {
        "n": len(per_case),
        "mean_score": sum(agg_scores) / len(agg_scores),
        "pass_rate": sum(1 for s in agg_scores if s >= 0.5) / len(agg_scores),
    }}
    path = tmp_path / "baseline.json"
    save_baseline(agg, path, per_case=per_case)
    return path, per_case


@pytest.mark.asyncio
async def test_compare_paired_no_change_does_not_regress(tmp_path):
    """Same per-case scores as baseline and current → gate must NOT fire."""
    prompts = {f"q{i}": [1.0 if i % 3 else 0.0] for i in range(8)}
    path, per_case = _build_baseline(tmp_path, prompts)
    # Current is identical to baseline.
    regressed, msgs = compare_to_baseline(
        aggregate_per_case(per_case), path, current_per_case=per_case,
    )
    assert regressed is False
    assert any("paired gate" in m for m in msgs)
    assert any("overall" in m for m in msgs)


@pytest.mark.asyncio
async def test_compare_paired_clear_regression_fires(tmp_path):
    """Every case drops pass→fail in current → gate MUST fire."""
    from evals.runner import PerCaseResult
    prompts = {f"q{i}": [1.0] for i in range(8)}
    path, base_per_case = _build_baseline(tmp_path, prompts)
    # Current: same case ids, all failing now.
    current = [
        PerCaseResult(
            case=pc.case, id=pc.id, tag=pc.tag, prompt=pc.prompt,
            scores=[0.0], answers=["x"],
        )
        for pc in base_per_case
    ]
    cur_agg = aggregate_per_case(current)
    regressed, msgs = compare_to_baseline(cur_agg, path, current_per_case=current)
    assert regressed is True
    assert any("REGRESSION" in m for m in msgs)


@pytest.mark.asyncio
async def test_compare_legacy_baseline_falls_back_with_warning(tmp_path, caplog):
    """A v1 baseline (no per-case data) uses the pp path even if current has
    per-case data — and logs that it predates paired gating."""
    import logging
    from evals.runner import PerCaseResult
    legacy = tmp_path / "legacy.json"
    save_baseline({"math": {"n": 4, "mean_score": 0.9, "pass_rate": 1.0}}, legacy)
    current = [
        PerCaseResult(
            case=EvalCase(prompt=f"q{i}", tag="math"),
            id=case_id("math", f"q{i}"), tag="math", prompt=f"q{i}",
            scores=[0.0], answers=["x"],
        )
        for i in range(4)
    ]
    with caplog.at_level(logging.WARNING):
        regressed, msgs = compare_to_baseline(
            aggregate_per_case(current), legacy, current_per_case=current,
        )
    # pass_rate 0% vs baseline 100% → -100pp → regression on the legacy gate.
    assert regressed is True
    assert any("predates paired gating" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_run_and_report_threads_repeats_and_reports_significance(tmp_path):
    # Fixtures
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2+2?", "tag": "math", "expected": 4}\n'
    )
    router = _StubRouter({"1+1?": "answer 2", "2+2?": "answer 4"})
    from evals.runner import run_and_report

    # First run: no baseline yet → save one with per-case data.
    report = await run_and_report(router, tmp_path, repeats=3)
    assert report["repeats"] == 3
    assert len(report["per_case"]) == 2
    assert all(len(pc["scores"]) == 3 for pc in report["per_case"])

    baseline = tmp_path / "baseline.json"
    save_baseline(
        aggregate_per_case(
            await run_eval_detailed(router, load_fixtures(tmp_path), repeats=3)
        ),
        baseline,
        per_case=await run_eval_detailed(router, load_fixtures(tmp_path), repeats=3),
    )
    report2 = await run_and_report(router, tmp_path, baseline, repeats=3)
    assert report2["gating_method"] == "paired"
    assert report2["regressed"] is False
    assert "significance" in report2
    assert "__overall__" in report2["significance"]
