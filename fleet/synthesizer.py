"""Pick the best response from parallel model outputs."""
from __future__ import annotations

import ast
import difflib
import re
from typing import Optional

from fleet.text import strip_thinking

# Cap inputs to expensive operations. ast.parse on multi-MB strings can
# trigger pathological recursion; difflib.SequenceMatcher is O(n²).
_MAX_AST_PARSE_CHARS = 200_000
_MAX_DIFF_CHARS = 4_000

# Match fenced code blocks: ```python ... ```, ``` ... ```, ```py ... ```
_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

ERROR_ALL_MODELS_FAILED = "(all models failed)"


def _extract_code(text: str) -> str:
    """Return the largest fenced code block, or the raw text if no fence."""
    matches = _FENCE_RE.findall(text)
    if not matches:
        return text
    return max(matches, key=len)


class Synthesizer:
    """Rule-based synthesis: no LLM, just heuristics."""

    MIN_SUMMARY_LENGTH = 20
    CONSENSUS_THRESHOLD = 0.3
    ALL_FAILED = ERROR_ALL_MODELS_FAILED

    def pick(
        self,
        responses: dict[str, Optional[str]],
        task_tag: str,
    ) -> str | dict[str, str]:
        """Return best response string, or a dict of all valid responses
        when there is no clear winner. Reasoning-model `<think>` blocks are
        stripped from every candidate before scoring AND before returning, so
        downstream consumers always see the final answer only."""
        valid: dict[str, str] = {}
        for k, v in responses.items():
            if not v:
                continue
            cleaned = strip_thinking(v)
            if cleaned:
                valid[k] = cleaned
        if not valid:
            return self.ALL_FAILED
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

    def _pick_code(self, valid: dict[str, str]) -> str:
        """Prefer responses whose fenced code block parses; longest valid wins."""
        valid_python = [
            text for text in valid.values() if self._is_valid_python(text)
        ]
        if valid_python:
            return max(valid_python, key=len)
        return max(valid.values(), key=len)

    @staticmethod
    def _is_valid_python(text: str) -> bool:
        # Strip prose by extracting the largest fenced block first; many model
        # outputs wrap code in ```python ... ``` and ast.parse on the wrapper
        # always fails.
        code = _extract_code(text)
        if len(code) > _MAX_AST_PARSE_CHARS:
            return False
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return False
        except Exception:  # noqa: BLE001 — defensive: ast can raise unexpected types
            return False
        return True

    def _pick_reasoning(self, valid: dict[str, str]) -> str | dict[str, str]:
        return self._consensus_or_longest(valid)

    def _pick_creative(self, valid: dict[str, str]) -> str:
        """Highest lexical diversity, longest as tie-break."""
        def diversity(text: str) -> float:
            words = text.split()
            if not words:
                return 0.0
            return len(set(words)) / len(words)
        return max(valid.values(), key=lambda t: (diversity(t), len(t)))

    def _pick_summarize(self, valid: dict[str, str]) -> str:
        """Shortest non-empty response (good summaries are concise)."""
        non_empty = [
            t for t in valid.values()
            if len(t.strip()) > self.MIN_SUMMARY_LENGTH
        ]
        if non_empty:
            return min(non_empty, key=len)
        return min(valid.values(), key=len)

    def _pick_general(self, valid: dict[str, str]) -> str | dict[str, str]:
        return self._consensus_or_longest(valid)

    def _consensus_or_longest(
        self, valid: dict[str, str]
    ) -> str | dict[str, str]:
        """Self-consistency score is primary. If consensus is weak
        (best_score < CONSENSUS_THRESHOLD), fall back to the longest response.
        Tie for longest returns the full ``valid`` dict so the user can choose.
        """
        items = list(valid.items())
        if len(items) < 2:
            return items[0][1]

        # Cap each text to bound the O(n²) SequenceMatcher cost on long
        # responses (~4000 chars × N²).
        truncated = [(name, text[:_MAX_DIFF_CHARS]) for name, text in items]

        scores: dict[str, float] = {}
        for i, (name, text_i) in enumerate(truncated):
            sims = [
                difflib.SequenceMatcher(None, text_i, text_j).ratio()
                for j, (_, text_j) in enumerate(truncated)
                if i != j
            ]
            scores[name] = sum(sims) / len(sims) if sims else 0.0

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score >= self.CONSENSUS_THRESHOLD:
            return valid[best_name]

        max_len = max(len(t) for t in valid.values())
        longest = [
            name for name, text in valid.items() if len(text) == max_len
        ]
        if len(longest) == 1:
            return valid[longest[0]]
        return valid
