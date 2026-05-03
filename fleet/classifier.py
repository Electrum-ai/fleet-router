"""Lightweight prompt classifier: keywords + optional embeddings."""
from __future__ import annotations

import re
from typing import Optional

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[misc]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


# Keyword maps: tag → list of compiled regex patterns
KEYWORD_MAP: dict[str, list[re.Pattern[str]]] = {
    "code": _compile([
        r"\bpython\b", r"\bjavascript\b", r"\bjs\b", r"\btypescript\b", r"\bts\b",
        r"\bfunction\b", r"\bclass\b", r"\brefactor\b", r"\bdebug\b", r"\berror\b",
        r"\bcompile\b", r"\bsyntax\b", r"\bscript\b", r"\bmodule\b", r"\bimport\b",
        r"\bwrite\b.*\bcode\b", r"\bgenerate\b.*\bcode\b", r"\bunit test\b",
        r"\btest\b.*\bfunction\b", r"\bapi\b.*\bendpoint\b",
    ]),
    "math": _compile([
        r"\bcalculate\b", r"\bsolve\b", r"\bequation\b", r"\bmath\b", r"\bstatistics\b",
        r"\bprobability\b", r"\bderivative\b", r"\bintegral\b", r"\bformula\b",
        r"\bsum\b.*\bnumbers\b", r"\bmultiply\b", r"\bdivide\b",
    ]),
    "reasoning": _compile([
        r"\bexplain\b.*\bwhy\b", r"\bcompare\b", r"\bcontrast\b", r"\banalyze\b",
        r"\bevaluate\b", r"\bpros?\b.*\bcons?\b", r"\badvantages?\b", r"\bdisadvantages?\b",
        r"\bwhat\b.*\bif\b", r"\bshould\b.*\bchoose\b",
    ]),
    "creative": _compile([
        r"\bpoem\b", r"\bstory\b", r"\bjoke\b", r"\btagline\b", r"\bslogan\b",
        r"\bcreative\b", r"\bimagine\b", r"\bdesign\b", r"\bbrainstorm\b",
        r"\bwrite\b.*\bstory\b", r"\bdraft\b", r"\bhook\b", r"\bcaption\b",
    ]),
    "summarize": _compile([
        r"\bsummarize\b", r"\bsummary\b", r"\btl;dr\b", r"\bkey points\b",
        r"\bmain ideas?\b", r"\brecap\b", r"\bcondense\b",
    ]),
    "translate": _compile([
        r"\btranslate\b", r"\btranslation\b", r"\bchinese\b", r"\barabic\b",
        r"\bspanish\b", r"\bfrench\b", r"\bgerman\b", r"\bjapanese\b",
    ]),
}

# Uncertainty keywords that trigger parallel mode
UNCERTAINTY_MARKERS: list[re.Pattern[str]] = _compile([
    r"\bbest\b", r"\bcompare\b", r"\breview\b", r"\bimprove\b",
    r"\boptimize\b", r"\bwhich\b.*\bbetter\b", r"\bwhat\b.*\bthink\b",
    r"\bsuggest\b", r"\brecommend\b",
])


class TaskClassifier:
    """Classify a prompt into a task tag and confidence score."""

    def __init__(self, embeddings_model: Optional[str] = None):
        self._embeddings_model = embeddings_model
        self._model = None
        self._tag_embeddings: Optional[dict] = None

        if embeddings_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(embeddings_model)
                # Precompute embeddings for each tag description
                self._tag_embeddings = {
                    tag: self._model.encode(f"Task: {tag}. {self._tag_description(tag)}")
                    for tag in KEYWORD_MAP
                }
            except Exception:
                # Graceful fallback if sentence-transformers unavailable
                self._model = None
                self._tag_embeddings = None

    @staticmethod
    def _tag_description(tag: str) -> str:
        descriptions = {
            "code": "writing programming code and software development",
            "math": "mathematical calculations and solving equations",
            "reasoning": "logical reasoning and analysis",
            "creative": "creative writing and storytelling",
            "summarize": "summarizing text and extracting key points",
            "translate": "translating between languages",
        }
        return descriptions.get(tag, "general task")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if np is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def classify(self, prompt: str) -> tuple[str, float]:
        prompt_lower = prompt.lower()

        # 1. Keyword scoring
        scores: dict[str, float] = {}
        for tag, patterns in KEYWORD_MAP.items():
            matches = sum(1 for p in patterns if p.search(prompt_lower))
            scores[tag] = min(matches * 0.85, 1.0)

        # 2. Uncertainty penalty
        uncertainty = sum(1 for p in UNCERTAINTY_MARKERS if p.search(prompt_lower))
        uncertainty_penalty = min(uncertainty * 0.15, 0.4)

        # 3. Embedding similarity (if available)
        if self._model and self._tag_embeddings:
            prompt_emb = self._model.encode(prompt)
            for tag, tag_emb in self._tag_embeddings.items():
                sim = self._cosine_similarity(prompt_emb, tag_emb)
                scores[tag] = max(scores[tag], sim * 0.8)

        # Pick best tag
        best_tag = max(scores, key=scores.get) if any(scores.values()) else "general"
        best_score = scores.get(best_tag, 0.0)

        # Apply uncertainty penalty
        confidence = max(0.0, best_score - uncertainty_penalty)

        return best_tag, confidence
