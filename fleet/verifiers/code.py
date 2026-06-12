"""Code verifier — AST validation + (optional) sandboxed execution.

Execution is HARD-GATED behind an operator-supplied sandbox. Candidate code
runs ONLY when BOTH `execute=True` AND a non-empty `sandbox` command template
is configured; in that case the code runs THROUGH that template (the project
never runs raw, unsandboxed code). If `execute=True` but no sandbox is set,
execution is DISABLED and the verifier falls back to AST-only static scoring,
emitting a one-time warning at construction.

The AST denylist (_DANGEROUS_IMPORTS/_DANGEROUS_CALLS) is an ADVISORY
pre-filter, NOT a security boundary: it is trivially bypassable (e.g.
``getattr(__builtins__, ...)``, ``().__class__.__subclasses__()``), so it
must never be relied on to make untrusted code safe to run. The sandbox
command (firejail/bubblewrap/Docker/etc.) is the only real boundary.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import shlex
import shutil
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

    def __init__(
        self,
        execute: bool = False,
        execute_timeout: int = 5,
        sandbox: str = "",
    ):
        self._execute = execute
        self._timeout = execute_timeout
        self._sandbox = (sandbox or "").strip()
        # Execution is gated on BOTH the opt-in AND a configured sandbox.
        self._will_execute = bool(execute and self._sandbox)
        if execute and not self._sandbox:
            logger.warning(
                "CodeVerifier: code_execute is ENABLED but no "
                "code_execute_sandbox is configured. The AST denylist is an "
                "ADVISORY pre-filter, NOT a security boundary — it is trivially "
                "bypassable (e.g. getattr(__builtins__, ...), "
                "().__class__.__subclasses__()). Code execution is therefore "
                "DISABLED; falling back to AST-only static scoring. Set "
                "synthesis.code_execute_sandbox to a sandbox command template "
                "(e.g. 'firejail --net=none --private={dir} {python} {file}') "
                "to run candidate code."
            )

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

        # Advisory pre-filter only — NOT a security boundary (see module
        # docstring). Rejects obviously-dangerous code before it would even
        # reach the sandbox, but the sandbox is what actually contains risk.
        dangerous, why = _has_dangerous_pattern(tree)
        if dangerous:
            # Dangerous-but-parseable code scores BELOW the static band floor
            # (0.30) so it can NEVER outrank a clean parseable candidate, and
            # below the default 0.4 abstention threshold so a lone dangerous
            # answer abstains by default rather than being returned. Still
            # strictly above the 0.0 syntax-error score (it does parse).
            # Ordering invariant maintained end-to-end:
            #   syntax_error 0.0 < dangerous 0.20 < static band [0.30, 0.58]
            #   < execution (runtime-error 0.6, clean 1.0).
            return candidate.with_score(
                0.20, f"parses; unsafe ({why}); not executing"
            )

        # Distributed scoring when execution is OFF (the default). The static
        # band is [0.30, 0.58]: a bare-but-parseable answer starts at 0.30 and
        # each discriminating signal adds 0.04. The band sits ENTIRELY BELOW
        # the execution band (runtime-error 0.6, clean 1.0) so an executed
        # candidate always outranks any static one, and ENTIRELY ABOVE the
        # dangerous-but-parseable sentinel (0.20) so flagged-dangerous code can
        # never outrank a clean candidate. AND — critically — a
        # bare-AST-valid-but-low-quality answer can now land below a calibrated
        # abstention threshold. The previous floor of 0.50 made any per-tag
        # code threshold below 0.50 unreachable, so abstention could never fire
        # on parseable-but-weak code; widening the band restores that lever.
        # AST-only scoring whenever execution is not actually enabled — either
        # execute=False, or execute=True with no configured sandbox (in which
        # case we DO NOT run the code; see the constructor warning).
        if not self._will_execute:
            score, notes = self._distributed_static_score(tree, code)
            return candidate.with_score(score, notes)

        # Execution gate — runs THROUGH the operator's sandbox with a timeout.
        exec_score, exec_note = await self._try_execute(code)
        # When execution is on, parse-only signals are noise; trust the
        # execution outcome.
        return candidate.with_score(exec_score, f"parses; {exec_note}")

    def _distributed_static_score(self, tree: ast.AST, code: str) -> tuple[float, str]:
        """0.30 base for parsing, +0.04 per discriminating signal, capped at
        0.58. Rationale for the [0.30, 0.58] band:

        - Floor 0.30 (not the old 0.50) lets a bare-AST-valid-but-low-quality
          answer (e.g. a `pass`-only stub) score below a calibrated abstention
          threshold, so per-tag code abstention can actually fire. The old
          0.50 floor made any cut-point below 0.50 a dead lever.
        - Ceiling 0.58 stays STRICTLY BELOW the execution band (runtime-error
          0.6, clean 1.0), so any executed candidate beats every static one —
          execution remains the dominant, trusted signal.
        - +0.04 per signal preserves the relative ordering (more signals ⇒
          higher score) so the bandit still gets graded, discriminative signal.

        A normal multi-signal answer (defines + returns + non-trivial body +
        reasonable length, ± documented/typed) lands ~0.42–0.58 — comfortably
        above the default 0.4 — so ordinary valid code does NOT abstain by
        default; only genuinely threadbare answers dip below."""
        # Conscious tradeoff: a short, imperative, print-only answer (e.g. a
        # 2-line script with no defs/returns) collects few signals and can land
        # below the default 0.4 threshold, so it abstains by default. We accept
        # that — correct one-liners are cheap to re-ask, and the false-abstain
        # cost is lower than returning unverifiable threadbare code as a winner.
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

        score = 0.30 + 0.04 * len(signals)
        score = min(score, 0.58)
        return score, "; ".join(["parses", *signals])

    def _build_sandbox_argv(self, file_path: str, workdir: str) -> list[str]:
        """Render the operator's sandbox command template into an argv list,
        substituting {python}/{file}/{dir}. Tokenized via shlex BEFORE
        substitution so a path containing spaces can't reshape the command."""
        subs = {
            "{python}": sys.executable,
            "{file}": file_path,
            "{dir}": workdir,
        }
        argv: list[str] = []
        for tok in shlex.split(self._sandbox):
            for placeholder, value in subs.items():
                tok = tok.replace(placeholder, value)
            argv.append(tok)
        return argv

    async def _try_execute(self, code: str) -> tuple[float, str]:
        # Write the candidate into a private temp dir so the sandbox template
        # has a {dir} to confine to. Execution always goes THROUGH the
        # operator's sandbox command — never a raw interpreter.
        workdir = tempfile.mkdtemp(prefix="fleet-code-")
        path = os.path.join(workdir, "candidate.py")
        try:
            with open(path, "w") as f:
                f.write(code)
            argv = self._build_sandbox_argv(path, workdir)
            if not argv:
                return 0.5, "exec scaffolding failed: empty sandbox command"
            proc = await asyncio.create_subprocess_exec(
                *argv,
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
                return 1.0, "executes cleanly (sandboxed)"
            err = stderr.decode("utf-8", errors="replace")[:200]
            return 0.6, f"runtime error: {err}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("code exec scaffolding failed: %s", exc)
            return 0.5, f"exec scaffolding failed: {type(exc).__name__}"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
