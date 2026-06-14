import pytest

from evals.compare import compare


class _StubRouter:
    def __init__(self, answers):
        self._answers = answers
    async def ask(self, prompt):
        return self._answers.get(prompt, "no answer")


@pytest.mark.asyncio
async def test_compare_reports_per_tag_deltas(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2+2?", "tag": "math", "expected": 4}\n'
    )
    router_a = _StubRouter({"1+1?": "the answer is 2", "2+2?": "I think 99"})
    router_b = _StubRouter({"1+1?": "the answer is 2", "2+2?": "the answer is 4"})

    report = await compare(
        a=("a", router_a),
        b=("b", router_b),
        fixtures_dir=tmp_path,
    )
    assert report["aggregates"]["a"]["math"]["pass_rate"] == 0.5
    assert report["aggregates"]["b"]["math"]["pass_rate"] == 1.0
    assert "delta=+50.0pp" in report["summary"]
    assert "→ b" in report["summary"]
    # New: a bootstrap CI on the per-case delta is reported and structured.
    assert "Mean delta (b - a)" in report["summary"]
    assert "CI[" in report["summary"]
    assert "bootstrap" in report
    assert report["bootstrap"]["overall"]["n"] == 2
    # b beat a on one of two cases (delta +1 and 0) → mean delta = +0.5.
    assert report["bootstrap"]["overall"]["delta"] == pytest.approx(0.5)
    assert "math" in report["bootstrap"]["per_tag"]


@pytest.mark.asyncio
async def test_compare_bootstrap_ci_is_deterministic(tmp_path):
    f = tmp_path / "math.jsonl"
    f.write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
        '{"prompt": "2+2?", "tag": "math", "expected": 4}\n'
    )
    ra = _StubRouter({"1+1?": "the answer is 2", "2+2?": "I think 99"})
    rb = _StubRouter({"1+1?": "the answer is 2", "2+2?": "the answer is 4"})
    r1 = await compare(a=("a", ra), b=("b", rb), fixtures_dir=tmp_path, seed=5)
    ra2 = _StubRouter({"1+1?": "the answer is 2", "2+2?": "I think 99"})
    rb2 = _StubRouter({"1+1?": "the answer is 2", "2+2?": "the answer is 4"})
    r2 = await compare(a=("a", ra2), b=("b", rb2), fixtures_dir=tmp_path, seed=5)
    assert r1["bootstrap"] == r2["bootstrap"]


@pytest.mark.asyncio
async def test_compare_handles_router_exception(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"prompt": "p", "tag": "math", "expected": 1}\n')

    class Crashing:
        async def ask(self, prompt):
            raise RuntimeError("network down")

    report = await compare(
        a=("crash", Crashing()),
        b=("ok", _StubRouter({"p": "1"})),
        fixtures_dir=tmp_path,
    )
    # Crash recorded as score=0 (extracted no number from error message)
    assert report["aggregates"]["crash"]["math"]["pass_rate"] == 0.0
    assert report["aggregates"]["ok"]["math"]["pass_rate"] == 1.0
