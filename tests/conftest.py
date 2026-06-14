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
"""

import pytest
import requests


@pytest.fixture(autouse=True)
def _no_live_ollama(monkeypatch):
    """Block real Ollama network calls for every test by default."""

    def _refuse(*args, **kwargs):
        raise requests.RequestException(
            "live network disabled in tests (see tests/conftest.py)"
        )

    monkeypatch.setattr("fleet.registry.requests.get", _refuse)
