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
    not run. Score caps at 0.5 (parses) without an execution bonus."""
    v = CodeVerifier(execute=True)
    candidates = [
        Candidate("a", 0, "import os\nos.system('echo hi')"),
    ]
    result = await v.aggregate("p", candidates)
    # Parses, but flagged as unsafe — score should be ≤ 0.5 (no exec bonus).
    assert result.winner is not None
    assert result.winner.score <= 0.5
    assert "unsafe" in result.winner.notes


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
    # AST-only static scoring caps at 0.85 (no execution bonus).
    assert result.winner.score <= 0.85
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
    assert result.winner.score <= 0.85
