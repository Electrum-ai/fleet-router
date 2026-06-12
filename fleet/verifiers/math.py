"""Math verifier — extracts numeric answers and majority-votes across candidates.

Self-consistency works particularly well here: when the same model is sampled
N times with temperature > 0, the majority numeric answer is usually correct
(Wang et al., 2022 — +18pp on GSM8K).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from fleet.text import strip_thinking
from fleet.verifiers.base import Candidate, VerificationResult

# A numeric token. Two alternatives, tried left to right:
#   1. comma-grouped integer/decimal: 1,234  /  1,234,567.5   (requires a comma
#      group, so it never swallows the first 3 digits of a plain integer)
#   2. plain integer/decimal with optional exponent: 42 / -3.5 / 1.5e3
_NUM = (
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"          # comma-grouped
    r"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"       # plain / scientific
)
# Optional simple-fraction tail: "/ 2", "/-3.5".
_FRACTION_TAIL = r"(?:\s*/\s*-?\d+(?:\.\d+)?)?"

# Explicit answer markers. The connector after the marker word allows an
# optional "is" ("the answer is 42"), or ":"/"=" ("answer: 42", "answer=42"),
# or nothing ("answer 42"). The capture supports comma grouping and simple
# fractions so the marker path and the fallback path agree.
_ANSWER_PATTERNS = [
    re.compile(
        r"(?:final\s+answer|answer|result|equals?|=)"
        r"\s*(?:is\s+|[:=]\s*)?\$?"
        r"((?:" + _NUM + r")" + _FRACTION_TAIL + r")",
        re.IGNORECASE,
    ),
    re.compile(r"\\boxed\{([^}]+)\}"),
]
# Fallback scanner: same numeric token as the marker path, including the
# optional fraction tail, so the fallback agrees with the marker on bare
# fractions ("we compute 1/2 here" -> "0.5", matching "x = 1/2"). The alternation
# in _NUM (A|B) MUST be grouped before the tail is appended — otherwise the tail
# binds only to the second alternative and "1/0" tokenizes as ["1", "0"], letting
# the bare denominator leak into the vote.
_NUMBER_RE = re.compile(r"(?:" + _NUM + r")" + _FRACTION_TAIL)
_FRACTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$")


def _normalize_float(f: float) -> str:
    """Canonicalize a float so '42', '42.0', '4.2e1' compare equal."""
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _to_number(token: str) -> Optional[str]:
    """Parse a numeric token — possibly comma-grouped or a simple a/b
    fraction — into a normalized canonical string, or None if it isn't a clean
    number.

    Fractions are evaluated to a normalized decimal (``1/2`` -> ``0.5``) so a
    candidate that writes a fraction votes as the SAME value as candidates that
    write the decimal — majority voting stays coherent. A malformed or
    zero-denominator fraction returns None (treated as non-numeric) rather than
    silently voting the numerator and corrupting the tally.
    """
    token = token.strip().rstrip(".").replace(",", "")
    if not token:
        return None
    frac = _FRACTION_RE.match(token)
    if frac:
        try:
            den = float(frac.group(2))
            if den == 0:
                return None
            return _normalize_float(float(frac.group(1)) / den)
        except (ValueError, OverflowError):
            return None
    try:
        return _normalize_float(float(token))
    except (ValueError, OverflowError):
        return None


def _extract_final_answer(text: str) -> Optional[str]:
    """Return the model's final numeric answer, normalized.

    Strategy: scan for explicit "answer is X" / \\boxed{X} markers first;
    fall back to the last number in the text (math models almost always
    end with their answer).
    """
    text = strip_thinking(text)
    if not text:
        return None
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(text):
            parsed = _to_number(m.group(1))
            if parsed is not None:
                return parsed
    # Fallback: scan every numeric token (comma- and fraction-aware) and return
    # the LAST one that parses cleanly. Skipping tokens _to_number rejects (e.g.
    # the zero-denominator "1/0") keeps the fallback honest about its own
    # docstring contract — a malformed fraction never votes its numerator or
    # denominator just because it sat at the end of the string.
    for tok in reversed(_NUMBER_RE.findall(text)):
        parsed = _to_number(tok)
        if parsed is not None:
            return parsed
    return None


class MathVerifier:
    """Score by numeric answer; pick by majority vote with size-aware tie-break."""

    tag = "math"

    async def aggregate(
        self,
        prompt: str,
        candidates: list[Candidate],
    ) -> VerificationResult:
        if not candidates:
            return VerificationResult(winner=None, all_scored=[], rationale="no candidates", abstain=True)

        # Extract answers; keep candidates without parseable answers at score 0.
        per_candidate: list[tuple[Candidate, Optional[str]]] = []
        for c in candidates:
            ans = _extract_final_answer(c.text)
            per_candidate.append((c, ans))

        valid_answers = [a for _, a in per_candidate if a is not None]
        if not valid_answers:
            scored = [c.with_score(0.0, "no numeric answer found") for c, _ in per_candidate]
            return VerificationResult(
                winner=None, all_scored=scored,
                rationale="no candidate produced a parseable numeric answer",
                abstain=True,
            )

        votes = Counter(valid_answers)
        winner_answer, winner_count = votes.most_common(1)[0]
        agreement = winner_count / len(valid_answers)

        scored: list[Candidate] = []
        for c, ans in per_candidate:
            if ans is None:
                scored.append(c.with_score(0.0, "no answer"))
            elif ans == winner_answer:
                scored.append(c.with_score(agreement, f"answer={ans} ({winner_count}/{len(valid_answers)} agree)"))
            else:
                scored.append(c.with_score(0.2, f"answer={ans} (disagrees with majority {winner_answer})"))

        # Pick winner: among candidates with the majority answer, prefer the longest
        # explanation (proxy for show-your-work quality).
        winners = [c for c in scored if c.score == agreement]
        winner = max(winners, key=lambda c: len(c.text))

        # Abstain on tied vote with no clear majority on >2 distinct answers.
        abstain = agreement < 0.5 and len(votes) > 1
        return VerificationResult(
            winner=None if abstain else winner,
            all_scored=scored,
            rationale=f"majority answer: {winner_answer} ({winner_count}/{len(valid_answers)})",
            abstain=abstain,
        )
