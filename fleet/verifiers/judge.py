"""LLM-as-judge verifier — sends candidates to a judge model for ranking.

Tag-specific rubrics make the judge focus on the right axes (faithfulness for
summarize, originality for creative, correctness for reasoning, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from fleet.providers.base import GenerateRequest, Provider
from fleet.verifiers.base import Candidate, VerificationResult

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_RUBRICS: dict[str, str] = {
    "code": "Score by correctness, edge-case handling, idiomatic style, and runnability.",
    "math": "Score by numeric correctness, completeness of working, and clarity of steps.",
    "reasoning": "Score by logical soundness, completeness of argument, and explicit handling of counterpoints.",
    "creative": "Score by originality, voice, coherence, and how well it answers the prompt.",
    "summarize": "Score by faithfulness to source, coverage of key points, and concision.",
    "translate": "Score by accuracy, fluency, and cultural appropriateness.",
    "general": "Score by accuracy, completeness, and helpfulness.",
}

_JUDGE_PROMPT = """You are evaluating LLM responses to a user prompt.

USER PROMPT:
{prompt}

EVALUATION CRITERIA: {rubric}

CANDIDATE RESPONSES:
{candidates}

For each candidate, give an integer score from 0 to 10. Then identify the best candidate by its label.

Reply with ONLY this JSON object — no prose before or after:
{{"scores": {{"A": 7, "B": 5}}, "best": "A", "rationale": "brief reason"}}"""


def _strip(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: find the first JSON object in text and parse it.

    String-aware: braces inside double-quoted strings (e.g. a rationale
    field that mentions `}`) are skipped so they don't unbalance the
    depth counter and corrupt the slice. Without this, judge outputs
    that include braces in their explanation silently fail to parse and
    poison the bandit with the all-0.5 fallback path."""
    text = _strip(text)
    # Try whole text first (model followed instructions).
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find a {...} block, ignoring braces inside string literals.
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = -1
                    continue
    return None


class JudgeVerifier:
    """Use an LLM to rank candidates against a tag-specific rubric.

    Falls back to first candidate if the judge call fails or returns
    unparseable output — preferable to crashing or silent abstention.
    """

    def __init__(
        self,
        provider: Provider,
        judge_model: str,
        tag: str = "general",
        temperature: float = 0.0,
    ):
        self._provider = provider
        self._model = judge_model
        self.tag = tag
        self._temperature = temperature

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)
        if len(candidates) == 1:
            # No judging happens — just pass it through. Mark unreliable
            # so the bandit doesn't bias toward this model just because
            # nobody else fielded a candidate this round.
            c = candidates[0].with_score(0.5, "only candidate; not judged")
            return VerificationResult(
                winner=c, all_scored=[c], rationale="only candidate",
                scores_reliable=False,
            )

        rubric = _RUBRICS.get(self.tag, _RUBRICS["general"])
        labels = [chr(65 + i) for i in range(len(candidates))]  # A, B, C, ...
        labeled = dict(zip(labels, candidates))

        candidates_block = "\n\n".join(
            f"--- Candidate {label} ---\n{_strip(c.text)}"
            for label, c in labeled.items()
        )
        judge_prompt = _JUDGE_PROMPT.format(
            prompt=prompt, rubric=rubric, candidates=candidates_block
        )
        req = GenerateRequest(
            model=self._model,
            prompt=judge_prompt,
            temperature=self._temperature,
            samples=1,
        )

        try:
            results = await self._provider.generate(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge provider crashed: %s", exc)
            results = []

        if not results or not results[0]:
            logger.warning("judge produced no output; falling back to first candidate")
            scored = [c.with_score(0.5, "judge unavailable") for c in candidates]
            return VerificationResult(
                winner=scored[0],
                all_scored=scored,
                rationale="judge unavailable; first candidate returned",
                scores_reliable=False,
            )

        parsed = _extract_json(results[0])
        if not parsed or not isinstance(parsed, dict):
            logger.warning("judge output not parseable as JSON")
            scored = [c.with_score(0.5, "judge output unparseable") for c in candidates]
            return VerificationResult(
                winner=scored[0],
                all_scored=scored,
                rationale="judge output unparseable; first candidate returned",
                scores_reliable=False,
            )

        raw_scores = parsed.get("scores", {})
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        best_label = parsed.get("best")
        rationale = str(parsed.get("rationale", ""))[:500]

        scored = []
        for label, c in labeled.items():
            raw = raw_scores.get(label)
            try:
                norm = max(0.0, min(1.0, float(raw) / 10.0))
            except (TypeError, ValueError):
                norm = 0.5
            scored.append(c.with_score(norm, f"judge: {raw}/10"))

        if best_label not in labeled:
            # Fallback to highest-scored.
            winner = max(scored, key=lambda c: c.score)
        else:
            best_idx = labels.index(best_label)
            winner = scored[best_idx]

        return VerificationResult(
            winner=winner,
            all_scored=scored,
            rationale=rationale or f"judge selected {best_label}",
        )
