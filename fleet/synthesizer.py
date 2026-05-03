"""Pick the best response from parallel model outputs."""
from __future__ import annotations

import difflib
import py_compile
import tempfile
from pathlib import Path
from typing import Optional


class Synthesizer:
    """Rule-based synthesis: no LLM, just heuristics."""

    def pick(self, responses: dict[str, Optional[str]], task_tag: str) -> str | dict[str, str]:
        """Return best response string, or dict of all if no clear winner."""
        # Filter out None/failed responses
        valid = {k: v for k, v in responses.items() if v}
        if not valid:
            return "(all models failed)"
        if len(valid) == 1:
            return next(iter(valid.values()))

        if task_tag == "code":
            return self._pick_code(valid)
        if task_tag in ("math", "reasoning"):
            return self._pick_reasoning(valid)
        if task_tag == "creative":
            return self._pick_creative(valid)
        if task_tag == "summarize":
            return self._pick_summarize(valid)
        return self._pick_general(valid)

    def _pick_code(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer syntactically valid Python."""
        for name, text in valid.items():
            if self._is_valid_python(text):
                return text
        # Fallback: longest
        return max(valid.values(), key=len)

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                path = f.name
            py_compile.compile(path, doraise=True)
            Path(path).unlink(missing_ok=True)
            return True
        except Exception:
            Path(path).unlink(missing_ok=True)
            return False

    def _pick_reasoning(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer consensus (similar answers)."""
        return self._consensus_or_longest(valid)

    def _pick_creative(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer longest response for creative tasks."""
        return max(valid.values(), key=len)

    def _pick_summarize(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Prefer shortest that still has content."""
        non_empty = [t for t in valid.values() if len(t.strip()) > 20]
        if non_empty:
            return min(non_empty, key=len)
        return min(valid.values(), key=len)

    def _pick_general(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Self-consistency: pick the answer most similar to others."""
        return self._consensus_or_longest(valid)

    def _consensus_or_longest(self, valid: dict[str, str]) -> str | dict[str, str]:
        """Pick the response with highest average similarity to others."""
        texts = list(valid.values())
        if len(texts) < 2:
            return texts[0]

        scores: dict[str, float] = {}
        for name, text in valid.items():
            sims = [
                difflib.SequenceMatcher(None, text, other).ratio()
                for other in texts if other != text
            ]
            scores[name] = sum(sims) / len(sims) if sims else 0.0

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        # If consensus is weak, return first valid response as fallback
        if best_score < 0.3:
            return next(iter(valid.values()))
        return valid[best_name]
