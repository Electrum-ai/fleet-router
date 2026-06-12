"""LLM-as-judge verifier — sends candidates to a judge model for ranking.

Tag-specific rubrics make the judge focus on the right axes (faithfulness for
summarize, originality for creative, correctness for reasoning, etc.).

De-biasing (PRIMARY scoring path — these scores feed the bandit):

- Self-preference (A): an LLM judge over-rates its own model's output. When
  the judge model is itself among the candidates, ``aggregate`` delegates to a
  NEUTRAL judge (a judge bound to a model that is NOT a candidate) supplied by
  ``neutral_factory``. With no neutral model available it judges as-is but says
  so in the rationale so the bias is visible downstream. In that no-neutral
  branch the scores are additionally marked ``scores_reliable=False`` — but
  ONLY when 2+ distinct candidate models are in play, i.e. there is genuine
  cross-model comparison that self-preference can skew. Single-pool
  self-consistency (all candidates are the judge's own single model) has no
  between-model preference to bias and the bandit has a single arm for that
  tag anyway, so those scores stay reliable.
- Position bias (B): LLM judges favor earlier/later slate positions. With
  ``swap_order=True`` (default) the judge runs TWICE — once in the given order,
  once reversed — and per-candidate normalized scores are averaged. When the
  two passes disagree on the single best candidate that is recorded in the
  rationale; the averaged top score itself drops toward the runner-up, which is
  the genuine low-confidence signal the abstention/escalation path reacts to.
- Slate hygiene (C): each candidate's text is truncated to a bounded length in
  the judge prompt (scoring/winner selection still map back to the FULL
  candidate), and labels use a base-26 scheme (A..Z, AA, AB, ...) that is
  collision-free past 26 candidates.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from fleet.providers.base import GenerateRequest, Provider
from fleet.text import strip_thinking
from fleet.verifiers.base import Candidate, VerificationResult

logger = logging.getLogger(__name__)

# Per-candidate text budget inside the judge prompt. Multi-sample configs put
# 9-21 long answers in front of the judge; without a bound the prompt blows past
# the context window and the tail candidates get silently dropped. Truncation is
# ONLY for the judge prompt — scoring and winner selection map back to the full
# Candidate object.
_MAX_CANDIDATE_CHARS = 3000

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


def _candidate_label(i: int) -> str:
    """Spreadsheet-style base-26 label for the i-th (0-based) candidate:
    0->A .. 25->Z, 26->AA, 27->AB, ... — collision-free for any count, unlike
    ``chr(65+i)`` which spills into '[', '\\', ']' at i>=26 (reachable with
    multi-sample configs, e.g. general:9 × 3 models = 27 candidates)."""
    label = ""
    n = i + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(65 + rem) + label
    return label


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CANDIDATE_CHARS:
        return text
    return text[:_MAX_CANDIDATE_CHARS] + "\n[truncated]"


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: find the first JSON object in text and parse it.

    String-aware: braces inside double-quoted strings (e.g. a rationale
    field that mentions `}`) are skipped so they don't unbalance the
    depth counter and corrupt the slice. Without this, judge outputs
    that include braces in their explanation silently fail to parse and
    poison the bandit with the all-0.5 fallback path."""
    text = strip_thinking(text)
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


@dataclass
class _Pass:
    """One judge pass over an ordered candidate list.

    `norms` is per-candidate normalized [0,1] scores aligned to the order the
    candidates were presented in (None when the pass failed). `best` is the
    index (into that same ordering) the judge named as best, or None when its
    label was missing/unknown. `reason` is empty on success, else the failure
    reason that becomes the fallback rationale (preserving the "unparseable"
    wording the bandit-poisoning regression test asserts on)."""

    norms: Optional[list[float]]
    best: Optional[int]
    rationale: str
    reason: str = ""


class JudgeVerifier:
    """Use an LLM to rank candidates against a tag-specific rubric.

    Falls back to first candidate (scores_reliable=False) if the judge call
    fails or returns unparseable output — preferable to crashing or silent
    abstention.
    """

    def __init__(
        self,
        provider: Provider,
        judge_model: str,
        tag: str = "general",
        temperature: float = 0.0,
        model_key: Optional[str] = None,
        neutral_factory: Optional[
            Callable[[set[str]], Optional["JudgeVerifier"]]
        ] = None,
        swap_order: bool = True,
    ):
        self._provider = provider
        # Provider-side model id used in GenerateRequest.
        self._model = judge_model
        # Registry key used to test self-preference against Candidate.model
        # (which carries registry keys). Defaults to the api model so callers
        # that pass a single name keep working; the router passes the real key.
        self._model_key = model_key if model_key is not None else judge_model
        self.tag = tag
        self._temperature = temperature
        self._neutral_factory = neutral_factory
        self._swap_order = swap_order

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

        # (A) Self-preference: if this judge's own model is among the
        # candidates, delegate to a neutral judge whose model is NOT a
        # candidate. If none exists, judge as-is but flag it in the rationale.
        self_pref_note = ""
        # Targeted reliability guard: when the judge grades its own model's
        # candidates with no neutral available AND 2+ distinct models are being
        # compared, the cross-model scores are self-preference-biased and must
        # NOT drive bandit posteriors. Single-model self-consistency stays
        # reliable (no between-model preference to bias).
        self_pref_unreliable = False
        candidate_models = {c.model for c in candidates}
        if self._model_key in candidate_models:
            neutral = (
                self._neutral_factory(candidate_models)
                if self._neutral_factory is not None
                else None
            )
            if neutral is not None:
                return await neutral.aggregate(prompt, candidates)
            self_pref_note = (
                "[self-preference: judge graded its own model's candidates; "
                "no neutral model available] "
            )
            self_pref_unreliable = len(candidate_models) > 1

        rubric = _RUBRICS.get(self.tag, _RUBRICS["general"])

        # (B) Position bias: pass 1 in the given order.
        pass1 = await self._judge_pass(prompt, rubric, candidates)
        if pass1.norms is None:
            return self._fallback(candidates, pass1.reason, self_pref_note)

        if not self._swap_order:
            return self._build_result(
                candidates, pass1.norms, pass1.best, pass1.rationale,
                self_pref_note, scores_reliable=not self_pref_unreliable,
            )

        # Pass 2 in reversed order, then map its results back to the canonical
        # candidate order and average per-candidate.
        reversed_cands = list(reversed(candidates))
        pass2 = await self._judge_pass(prompt, rubric, reversed_cands)
        if pass2.norms is None:
            # The swap pass failed; degrade to single-pass scores rather than
            # discarding a perfectly good first pass.
            note = self_pref_note + "[swap-order pass failed; single-pass scores] "
            return self._build_result(
                candidates, pass1.norms, pass1.best, pass1.rationale, note,
                scores_reliable=not self_pref_unreliable,
            )

        n = len(candidates)
        norms2 = list(reversed(pass2.norms))  # back to canonical order
        best2 = (n - 1 - pass2.best) if pass2.best is not None else None
        averaged = [(pass1.norms[i] + norms2[i]) / 2.0 for i in range(n)]

        note = self_pref_note
        if (
            pass1.best is not None
            and best2 is not None
            and pass1.best != best2
        ):
            note += (
                f"[swap-order disagreement: forward best={_candidate_label(pass1.best)}, "
                f"reversed best={_candidate_label(best2)}; averaged] "
            )
        rationale = note + (pass1.rationale or "judge averaged over swap order")
        return self._build_averaged_result(
            candidates, averaged, rationale,
            scores_reliable=not self_pref_unreliable,
        )

    async def _judge_pass(
        self, prompt: str, rubric: str, ordered: list[Candidate]
    ) -> _Pass:
        """Run ONE judge call over `ordered`, returning normalized scores
        aligned to that ordering plus the judge's named-best index."""
        labels = [_candidate_label(i) for i in range(len(ordered))]
        candidates_block = "\n\n".join(
            f"--- Candidate {label} ---\n{_truncate(strip_thinking(c.text))}"
            for label, c in zip(labels, ordered)
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
            logger.warning("judge produced no output")
            return _Pass(None, None, "", "judge unavailable")

        parsed = _extract_json(results[0])
        if not parsed or not isinstance(parsed, dict):
            logger.warning("judge output not parseable as JSON")
            return _Pass(None, None, "", "judge output unparseable")

        raw_scores = parsed.get("scores", {})
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        norms: list[float] = []
        for label in labels:
            raw = raw_scores.get(label)
            try:
                norms.append(max(0.0, min(1.0, float(raw) / 10.0)))
            except (TypeError, ValueError):
                norms.append(0.5)
        best_label = parsed.get("best")
        best = labels.index(best_label) if best_label in labels else None
        rationale = str(parsed.get("rationale", ""))[:500]
        return _Pass(norms, best, rationale)

    def _build_result(
        self,
        candidates: list[Candidate],
        norms: list[float],
        best: Optional[int],
        rationale: str,
        note: str,
        scores_reliable: bool = True,
    ) -> VerificationResult:
        """Single-pass result: winner is the judge's named best (or, when that
        label was missing, the highest-scored candidate) — preserving the
        original behavior when swap-order is OFF.

        ``scores_reliable`` is threaded from ``aggregate`` so the cross-model
        self-preference case (judge graded its own model, no neutral, 2+
        distinct models) can be marked unreliable and excluded from the
        bandit, while keeping the self-pref rationale note visible."""
        scored = [
            c.with_score(norm, f"judge: {round(norm * 10)}/10")
            for c, norm in zip(candidates, norms)
        ]
        if best is None:
            winner = max(scored, key=lambda c: c.score)
            chosen = f"highest-scored {winner.model}"
        else:
            winner = scored[best]
            chosen = _candidate_label(best)
        return VerificationResult(
            winner=winner,
            all_scored=scored,
            rationale=note + (rationale or f"judge selected {chosen}"),
            scores_reliable=scores_reliable,
        )

    def _build_averaged_result(
        self,
        candidates: list[Candidate],
        averaged: list[float],
        rationale: str,
        scores_reliable: bool = True,
    ) -> VerificationResult:
        """Swap-order result: winner is argmax of the AVERAGED scores.

        ``scores_reliable`` is threaded from ``aggregate`` so the cross-model
        self-preference case stays out of the bandit even with swap-order ON
        (position de-biasing does not remove self-preference bias)."""
        scored = [
            c.with_score(avg, f"judge: {avg * 10:.1f}/10 (swap-avg)")
            for c, avg in zip(candidates, averaged)
        ]
        winner = max(scored, key=lambda c: c.score)
        return VerificationResult(
            winner=winner, all_scored=scored, rationale=rationale,
            scores_reliable=scores_reliable,
        )

    def _fallback(
        self, candidates: list[Candidate], reason: str, note: str
    ) -> VerificationResult:
        """Judge crashed / empty / unparseable: return the first candidate with
        scores marked UNRELIABLE so the bandit never ingests judge-failure
        noise (the 0.5 all-equal fallback)."""
        scored = [c.with_score(0.5, reason) for c in candidates]
        return VerificationResult(
            winner=scored[0],
            all_scored=scored,
            rationale=note + f"{reason}; first candidate returned",
            scores_reliable=False,
        )
