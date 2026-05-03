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
    clf = TaskClassifier()
    tag, conf = clf.classify(prompt)
    assert tag == expected_tag
    assert conf >= 0.85


def test_keyword_classify_no_match_is_low_confidence():
    clf = TaskClassifier()
    tag, conf = clf.classify("do something nice")
    assert tag == "general"
    assert conf < 0.8


def test_uncertainty_penalty():
    clf = TaskClassifier()
    tag, conf = clf.classify("which is better: python or ruby")
    assert tag == "code"
    assert conf < 0.85


def test_embedding_path():
    clf = TaskClassifier(embeddings_model="all-MiniLM-L6-v2")
    tag, conf = clf.classify("write a python function to sort a list")
    assert tag == "code"
    assert conf >= 0.0


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
        assert conf >= 0.85
