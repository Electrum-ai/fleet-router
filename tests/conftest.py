"""Shared test fixtures.

The most important job here is network isolation: ``ModelRegistry.refresh``
performs a live ``GET /api/tags`` against a running Ollama daemon. Without a
guard, any test that constructs a registry and does *not* pre-populate
``_available`` (or pre-populates it with an empty set) silently reaches out to
whatever Ollama happens to be running on the host machine. That made
``test_cloud_models_in_all_available`` pass or fail depending on the developer's
local models — a non-hermetic test.

This autouse fixture neutralizes the live fetch for the whole suite by making
the network boundary (``fleet.registry.requests.get``) raise a connection
error by default, which ``refresh`` already handles by yielding an empty
registry. Tests that exercise refresh behavior re-patch ``requests.get``
themselves via ``@patch`` — that decorator applies inside the test body and
cleanly overrides this default. Tests that need installed-model state set it
explicitly via ``registry.set_available(...)``.

A second autouse fixture mocks ``SentenceTransformer`` so tests that construct
a ``FleetRouter`` (and thus a ``TaskClassifier``) don't trigger a HuggingFace
model download. Tests that specifically exercise the embeddings path use
``pytest.importorskip`` + a marker to opt out of the mock.
"""

import pytest
import requests
from unittest.mock import MagicMock, patch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_embeddings: use real SentenceTransformer (needs HF cache)"
    )


@pytest.fixture(autouse=True)
def _no_live_ollama(monkeypatch):
    """Block real Ollama network calls for every test by default."""

    def _refuse(*args, **kwargs):
        raise requests.RequestException(
            "live network disabled in tests (see tests/conftest.py)"
        )

    monkeypatch.setattr("fleet.registry.requests.get", _refuse)


@pytest.fixture(autouse=True)
def _mock_sentence_transformer(request, monkeypatch):
    """Mock SentenceTransformer so FleetRouter construction doesn't trigger
    a HuggingFace model download in CI / sandboxes.

    Tests that specifically exercise the real embeddings code path use
    ``@pytest.mark.real_embeddings`` to opt out of this mock. Those tests
    should also use ``pytest.importorskip("sentence_transformers")`` to
    skip when the package isn't installed.
    """
    if "real_embeddings" in request.keywords:
        return  # Let the real SentenceTransformer load

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock()

    class _MockST:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, text, *args, **kwargs):
            return MagicMock()

    # Patch sys.modules so `from sentence_transformers import SentenceTransformer`
    # inside TaskClassifier.__init__ picks up the mock.
    import sys
    mock_module = MagicMock()
    mock_module.SentenceTransformer = _MockST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_module)
