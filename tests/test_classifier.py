from unittest.mock import patch

import pytest

from fleet.classifier import TaskClassifier


@pytest.mark.parametrize(
    "prompt,expected_tag",
    [
        ("write a python function to sort a list", "code"),
        ("write a poem about the ocean", "creative"),
        ("solve this equation: 2x + 5 = 13", "math"),
        ("explain why the sky is blue", "reasoning"),
        ("summarize this article in three sentences", "summarize"),
        ("translate hello to japanese", "translate"),
    ],
)
def test_keyword_classify(prompt, expected_tag):
    """Tag selection is correct; confidence is positive."""
    clf = TaskClassifier()
    tag, conf = clf.classify(prompt)
    assert tag == expected_tag
    assert conf > 0.0


def test_keyword_classify_strong_match_clears_threshold():
    """A multi-keyword prompt should confidently route to a single model."""
    clf = TaskClassifier()
    # Hits: \bpython\b, \bfunction\b, \bwrite\b.*\bcode\b, \berror\b → 4 matches
    tag, conf = clf.classify(
        "write python code: a function that handles errors"
    )
    assert tag == "code"
    assert conf >= 0.8


def test_keyword_classify_single_match_below_threshold():
    """A single accidental keyword must not trip the single-model threshold."""
    clf = TaskClassifier()
    tag, conf = clf.classify("I had an error yesterday and felt sad")
    # 'error' matches 'code' but only weakly.
    assert conf < 0.8


def test_keyword_classify_no_match_is_low_confidence():
    clf = TaskClassifier()
    tag, conf = clf.classify("do something nice")
    assert tag == "general"
    assert conf < 0.8


def test_uncertainty_penalty():
    clf = TaskClassifier()
    tag, conf = clf.classify("which is better: python or ruby")
    # 'python' triggers code; 'best/which...better' add uncertainty penalty.
    assert tag == "code"
    assert conf < 0.8


@pytest.mark.real_embeddings
def test_embedding_path():
    """Verify the embedding code path actually runs (not the keyword fallback).

    Skipped when sentence-transformers is not installed in the test env."""
    pytest.importorskip("sentence_transformers")
    clf = TaskClassifier(embeddings_model="all-MiniLM-L6-v2")
    assert clf._model is not None
    assert clf._tag_embeddings is not None
    tag, conf = clf.classify("write a python function to sort a list")
    assert tag == "code"
    assert conf > 0.0


@pytest.mark.real_embeddings
def test_embedding_fallback_on_import_error():
    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        clf = TaskClassifier(embeddings_model="all-MiniLM-L6-v2")
        assert clf._model is None
        assert clf._tag_embeddings is None
        tag, conf = clf.classify("write a python function to sort a list")
        assert tag == "code"
        assert conf > 0.0
