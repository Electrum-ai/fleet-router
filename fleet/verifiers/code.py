"""Code verifier — AST validation + (optional) sandboxed execution.

Execution is OPT-IN because running arbitrary LLM-generated code is a real
RCE vector. Even with `execute=True`, we statically reject obvious dangerous
patterns (subprocess, os.system, network, file I/O) before running. This is
NOT a sandbox — for production use, wrap in firejail/bubblewrap/Docker.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import sys
import tempfile

from fleet.text import strip_thinking
from fleet.verifiers.base import Candidate, VerificationResult, Verifier

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Modules / names blocklisted from execution. Conservative — false positives
# are fine here, false negatives leak RCE.
_DANGEROUS_IMPORTS = {
    "subprocess", "os", "sys", "socket", "urllib", "requests", "httpx",
    "shutil", "ctypes", "multiprocessing", "asyncio", "pathlib",
    "importlib", "tempfile", "pickle", "marshal", "pty", "fcntl",
}
_DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
}


def _extract_code(text: str) -> str:
    matches = _FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len)
    return text


def _has_dangerous_pattern(tree: ast.AST) -> tuple[bool, str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in _DANGEROUS_IMPORTS:
                    return True, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in _DANGEROUS_IMPORTS:
                return True, f"from {node.module} import"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _DANGEROUS_CALLS:
                return True, f"call to {f.id}"
            if isinstance(f, ast.Attribute) and f.attr in _DANGEROUS_CALLS:
                return True, f"call to .{f.attr}"
    return False, ""


class CodeVerifier:
    """Score code candidates by AST validity and (optional) execution."""

    tag = "code"

    def __init__(self, execute: bool = False, execute_timeout: int = 5):
        self._execute = execute
        self._timeout = execute_timeout

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)

        scored: list[Candidate] = []
        for c in candidates:
            scored.append(await self._score_one(c))

        winner = max(scored, key=lambda c: (c.score, len(c.text)))
        # Code is mostly objective: if even one parses, return it. Only
        # abstain when nothing even compiles.
        abstain = winner.score == 0.0
        return VerificationResult(
            winner=winner if not abstain else None,
            all_scored=scored,
            rationale=winner.notes,
            abstain=abstain,
        )

    async def _score_one(self, candidate: Candidate) -> Candidate:
        code = _extract_code(strip_thinking(candidate.text))
        if not code.strip():
            return candidate.with_score(0.0, "no code found")

        # Static checks — base score.
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            return candidate.with_score(0.0, f"syntax error: {exc}")
        except (RecursionError, MemoryError):
            return candidate.with_score(0.0, "parse exhausted resources")

        dangerous, why = _has_dangerous_pattern(tree)
        if dangerous:
            return candidate.with_score(0.5, f"parses; unsafe ({why}); not executing")

        # Distributed scoring when execution is OFF (the default). The
        # previous implementation gave all-parse-with-defs candidates the
        # same ~0.7 score, which (a) collapsed bandit signal and (b) sat
        # exactly above the escalation threshold so EVERY code prompt
        # triggered escalation. Spreading multiple weak signals across
        # 0.50–0.85 lets candidates differentiate while keeping room for
        # an "executes cleanly" candidate to win at 1.0 when execute=True.
        if not self._execute:
            score, notes = self._distributed_static_score(tree, code)
            return candidate.with_score(score, notes)

        # Execution gate — runs in a subprocess with timeout.
        exec_score, exec_note = await self._try_execute(code)
        # When execution is on, parse-only signals are noise; trust the
        # execution outcome.
        return candidate.with_score(exec_score, f"parses; {exec_note}")

    def _distributed_static_score(self, tree: ast.AST, code: str) -> tuple[float, str]:
        """0.50 base for parsing, +0.05 per discriminating signal, capped
        at 0.85 so an executes-cleanly candidate can still beat any
        static winner. Goal: spread candidates across the score range so
        the bandit gets real signal AND escalation only fires for
        genuine uncertainty (not on every code prompt)."""
        signals: list[str] = []

        nodes = list(ast.walk(tree))
        defs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

        if defs:
            signals.append(f"defines {len(defs)} symbol(s)")
        if ast.get_docstring(tree) is not None or any(
            ast.get_docstring(d) is not None for d in defs
        ):
            signals.append("documented")
        # Type hints — strong correlation with intentional, careful code.
        if any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (n.returns is not None
                 or any(a.annotation is not None for a in n.args.args))
            for n in nodes
        ):
            signals.append("typed")
        # Has at least one return — distinguishes "function with body"
        # from "function with pass".
        if any(isinstance(n, ast.Return) and n.value is not None for n in nodes):
            signals.append("returns value")
        # Self-tests / assertions — strong intentionality signal.
        if any(isinstance(n, ast.Assert) for n in nodes):
            signals.append("self-tests")
        # Pass-only body is a near-certain non-answer.
        non_trivial = not all(
            len(d.body) == 1 and isinstance(d.body[0], ast.Pass)
            for d in defs
        ) if defs else True
        if non_trivial:
            signals.append("non-trivial body")
        # Reasonable line count — too short = stub, too long = noise.
        line_count = code.count("\n") + 1
        if 3 <= line_count <= 200:
            signals.append("reasonable length")

        score = 0.50 + 0.05 * len(signals)
        score = min(score, 0.85)
        return score, "; ".join(["parses", *signals])

    async def _try_execute(self, code: str) -> tuple[float, str]:
        path: str = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                path = f.name
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 0.5, "execution timed out"
            if proc.returncode == 0:
                return 1.0, "executes cleanly"
            err = stderr.decode("utf-8", errors="replace")[:200]
            return 0.6, f"runtime error: {err}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("code exec scaffolding failed: %s", exc)
            return 0.5, f"exec scaffolding failed: {type(exc).__name__}"
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
