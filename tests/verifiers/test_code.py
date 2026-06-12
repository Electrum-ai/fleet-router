import os
import sys

import pytest

from fleet.verifiers.base import Candidate
from fleet.verifiers.code import CodeVerifier


@pytest.mark.asyncio
async def test_code_verifier_picks_parseable_over_garbage():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "this is not python ("),
        Candidate("b", 0, "def f():\n    return 1"),
    ]
    result = await v.aggregate("prompt", candidates)
    assert result.winner is not None
    assert result.winner.model == "b"
    assert not result.abstain


@pytest.mark.asyncio
async def test_code_verifier_strips_fenced_blocks():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "Here is the code:\n```python\ndef foo():\n    return 42\n```\nLet me know."),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert "def foo" in result.winner.text


@pytest.mark.asyncio
async def test_code_verifier_abstains_when_nothing_parses():
    v = CodeVerifier()
    candidates = [
        Candidate("a", 0, "definitely not code ((("),
        Candidate("b", 0, "more not code )))"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.abstain
    assert result.winner is None


@pytest.mark.asyncio
async def test_code_verifier_refuses_to_execute_dangerous_code():
    """Even with execute=True, statically-detectable dangerous patterns must
    not run. The dangerous-but-parseable sentinel scores 0.20 — BELOW the
    static band floor (0.30) and below the default 0.4 abstention threshold —
    so flagged code can never outrank clean code and abstains by default."""
    v = CodeVerifier(execute=True)
    candidates = [
        Candidate("a", 0, "import os\nos.system('echo hi')"),
    ]
    result = await v.aggregate("p", candidates)
    # Parses, but flagged as unsafe — score is the 0.20 sentinel, strictly
    # below the static band floor (0.30) and the default threshold (0.4).
    assert result.winner is not None
    assert result.winner.score <= 0.2
    assert result.winner.score < 0.3
    assert "unsafe" in result.winner.notes


@pytest.mark.asyncio
async def test_dangerous_code_never_outranks_clean_code():
    """Regression (audit BLOCKER): flagged-dangerous code must NOT win against
    ordinary clean code. Previously the dangerous sentinel was 0.50 — near the
    TOP of the static band — so `import os; os.system(...)` (0.50) beat a clean
    `def add` (~0.42) and was returned as the winner. With the sentinel dropped
    to 0.20 the CLEAN candidate wins."""
    v = CodeVerifier()
    clean = Candidate("clean", 0, "def add(a, b):\n    return a + b")
    dangerous = Candidate("dangerous", 0, "import os\nos.system('rm -rf /')")
    result = await v.aggregate("p", [clean, dangerous])
    assert result.winner is not None
    assert result.winner.model == "clean"
    scores = {c.model: c.score for c in result.all_scored}
    assert scores["dangerous"] == 0.20
    assert scores["clean"] > scores["dangerous"]
    assert not result.abstain


@pytest.mark.asyncio
async def test_lone_dangerous_candidate_scores_below_default_threshold():
    """A lone dangerous candidate scores below the default 0.4 abstention
    threshold (and below the static band floor 0.30), so the synthesizer's
    threshold overlay abstains rather than returning unsafe code."""
    v = CodeVerifier()
    result = await v.aggregate("p", [Candidate("a", 0, "import socket\nsocket.socket()")])
    assert result.winner is not None
    assert result.winner.score == 0.20
    assert result.winner.score < 0.3   # below static band floor
    assert result.winner.score < 0.4   # below default abstention threshold


@pytest.mark.asyncio
async def test_code_score_ordering_invariant():
    """End-to-end ordering invariant:
        syntax_error 0.0 < dangerous 0.20 < static band [0.30, 0.58]
        < execution (clean 1.0).
    """
    # Static (execution off) tier: syntax error, dangerous, clean static.
    static_v = CodeVerifier()
    syntax_err = (await static_v.aggregate(
        "p", [Candidate("x", 0, "def broken( ((")]
    )).all_scored[0].score
    dangerous = (await static_v.aggregate(
        "p", [Candidate("x", 0, "import os\nos.system('x')")]
    )).all_scored[0].score
    clean_static = (await static_v.aggregate(
        "p", [Candidate("x", 0, "def add(a, b):\n    return a + b")]
    )).all_scored[0].score

    # Execution tier (sandbox just runs the interpreter): clean executable.
    exec_v = CodeVerifier(execute=True, execute_timeout=10, sandbox="{python} -I {file}")
    clean_exec = (await exec_v.aggregate(
        "p", [Candidate("x", 0, "x = 1 + 1\nassert x == 2")]
    )).all_scored[0].score

    assert syntax_err == 0.0
    assert dangerous == 0.20
    assert 0.30 <= clean_static <= 0.58
    assert syntax_err < dangerous < clean_static < clean_exec
    assert clean_exec == 1.0


@pytest.mark.asyncio
async def test_code_verifier_executes_safe_code_with_sandbox():
    """Execution is gated on a configured sandbox. Here the 'sandbox' template
    just runs the interpreter directly — enough to exercise the substitution +
    execution path and confirm clean code scores 1.0."""
    v = CodeVerifier(execute=True, execute_timeout=10, sandbox="{python} -I {file}")
    candidates = [
        Candidate("a", 0, "x = 1 + 1\nassert x == 2"),
    ]
    result = await v.aggregate("p", candidates)
    assert result.winner is not None
    assert result.winner.score == 1.0
    assert "executes cleanly" in result.winner.notes


@pytest.mark.asyncio
async def test_code_verifier_handles_empty_input():
    v = CodeVerifier()
    result = await v.aggregate("p", [])
    assert result.abstain


# ---------- C1: code execution is hard-gated behind a sandbox ----------


@pytest.mark.asyncio
async def test_code_execute_without_sandbox_never_executes(caplog):
    """execute=True but no sandbox configured: code must NOT run (no
    subprocess spawned), a warning is emitted at construction, and scoring
    falls back to AST-only static signals."""
    import logging
    from unittest.mock import patch

    with caplog.at_level(logging.WARNING, logger="fleet.verifiers.code"):
        v = CodeVerifier(execute=True)  # no sandbox
    # Construction warns loudly about the disabled execution.
    assert any(
        "code_execute is ENABLED but no" in rec.message
        and "DISABLED" in rec.message
        for rec in caplog.records
    )

    with patch(
        "fleet.verifiers.code.asyncio.create_subprocess_exec"
    ) as spawn:
        result = await v.aggregate("p", [Candidate("a", 0, "def f():\n    return 1")])

    # Never spawned a subprocess.
    spawn.assert_not_called()
    assert result.winner is not None
    # AST-only static scoring stays within the static band [0.30, 0.58], which
    # sits strictly BELOW the execution band (0.6 runtime-error / 1.0 clean) so
    # an executed candidate always outranks a static one. (Was <=0.85 before
    # the band was widened to make per-tag code abstention thresholds reachable.)
    assert result.winner.score < 0.6
    assert "executes cleanly" not in result.winner.notes


@pytest.mark.asyncio
async def test_code_execute_with_sandbox_builds_substituted_argv():
    """execute=True + sandbox template: candidate runs THROUGH the operator's
    sandbox command with {python}/{file}/{dir} substituted — assert the argv
    handed to create_subprocess_exec, without actually running firejail."""
    from unittest.mock import AsyncMock, patch

    template = "firejail --net=none --private={dir} {python} {file}"
    v = CodeVerifier(execute=True, sandbox=template)

    captured: dict = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

    async def _fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        return _FakeProc()

    with patch(
        "fleet.verifiers.code.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=_fake_exec),
    ):
        result = await v.aggregate("p", [Candidate("a", 0, "x = 1\nassert x == 1")])

    argv = captured["argv"]
    assert argv[0] == "firejail"
    assert "--net=none" in argv
    assert sys.executable in argv
    # {dir} and {file} were substituted to a real temp path, not left literal.
    assert not any("{dir}" in a or "{file}" in a or "{python}" in a for a in argv)
    private_arg = next(a for a in argv if a.startswith("--private="))
    file_arg = argv[-1]
    assert private_arg.endswith(os.path.dirname(file_arg))
    assert file_arg.endswith("candidate.py")
    assert result.winner is not None
    assert result.winner.score == 1.0


@pytest.mark.asyncio
async def test_code_execute_false_does_not_execute():
    """execute=False (default) is unchanged: no subprocess, AST-only scoring."""
    from unittest.mock import patch

    v = CodeVerifier(execute=False)
    with patch("fleet.verifiers.code.asyncio.create_subprocess_exec") as spawn:
        result = await v.aggregate("p", [Candidate("a", 0, "def f():\n    return 1")])
    spawn.assert_not_called()
    assert result.winner is not None
    # Static band ceiling is 0.58 (below the execution band). Was <=0.85.
    assert result.winner.score < 0.6


# ---------- B: widened static band [0.30, 0.58] ----------


@pytest.mark.asyncio
async def test_static_band_bare_stub_scores_below_default_threshold():
    """A pass-only stub is a near-non-answer: it must land BELOW the default
    0.4 abstention threshold so per-tag code abstention can fire on weak code.
    Under the old 0.50 floor this was impossible."""
    v = CodeVerifier()
    result = await v.aggregate("p", [Candidate("a", 0, "def f():\n    pass")])
    assert result.winner is not None
    # defines(1 symbol) is the only signal → 0.30 + 0.04 = 0.34.
    assert result.winner.score < 0.4
    # The verifier itself only abstains on a 0.0 score; the threshold overlay
    # lives in the synthesizer. Here we just assert the band placement.
    assert not result.abstain


@pytest.mark.asyncio
async def test_static_band_rich_answer_clears_default_threshold():
    """A normal multi-signal answer must stay comfortably above 0.4 so ordinary
    valid code does NOT abstain by default."""
    v = CodeVerifier()
    rich = (
        'def solve(n: int) -> int:\n'
        '    """Double n."""\n'
        '    result = n * 2\n'
        '    assert result == n + n\n'
        '    return result\n'
    )
    result = await v.aggregate("p", [Candidate("a", 0, rich)])
    assert result.winner is not None
    assert result.winner.score >= 0.4
    # Still capped strictly below the execution band.
    assert result.winner.score <= 0.58


@pytest.mark.asyncio
async def test_static_band_preserves_signal_ordering():
    """More discriminating signals ⇒ strictly higher score (the band keeps
    bandit signal). Rich answer > bare stub, both within [0.30, 0.58]."""
    v = CodeVerifier()
    stub = Candidate("stub", 0, "def f():\n    pass")
    rich = Candidate(
        "rich", 0,
        'def g(x: int) -> int:\n    """doc."""\n    return x + 1\n',
    )
    result = await v.aggregate("p", [stub, rich])
    by_model = {c.model: c.score for c in result.all_scored}
    assert by_model["rich"] > by_model["stub"]
    assert all(0.30 <= s <= 0.58 for s in by_model.values())
    assert result.winner.model == "rich"
