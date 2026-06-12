"""Wiring tests for two previously-inert config blocks:

- classifier.mode == "llm"  → FleetRouter routes classification through
  LLMClassifier (provider resolved from the pool) instead of the sync
  keyword TaskClassifier.
- retrieval.enabled          → FleetRouter prepends retrieved context to the
  DISPATCHED prompt for configured tags, while classification still sees the
  ORIGINAL prompt.

The components themselves (LLMClassifier, retrieval providers) are unit-tested
elsewhere — these tests assert the wiring: that the flags take effect and that
defaults stay byte-for-byte unchanged.
"""
from unittest.mock import AsyncMock, patch

import pytest

from fleet.config import (
    ClassifierConfig,
    Config,
    ModelEntry,
    RetrievalConfig,
    SamplingConfig,
    SynthesisConfig,
    ThresholdConfig,
)
from fleet.llm_classifier import LLMClassifier
from fleet.providers.pool import ProviderPool
from fleet.retrieval import NoOpRetrieval, WebSearchRetrieval
from fleet.router import FleetRouter


# Legacy single-call knobs: heuristic synthesis + a clearable confidence bar +
# 1 sample per tag keep these tests on the dispatcher.run code path so we can
# assert against the prompt that reaches the dispatcher directly.
def _legacy(**overrides):
    base = dict(
        synthesis=SynthesisConfig(mode="heuristic"),
        thresholds=ThresholdConfig(single_confidence=0.8),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
    )
    base.update(overrides)
    return Config(**base)


def _make_router(config, stub_provider=None):
    """Build a FleetRouter, optionally injecting a stub provider into the pool
    under the name "ollama" so the LLM classifier resolves to it."""
    if stub_provider is None:
        return FleetRouter(config)
    pool = ProviderPool({"ollama": stub_provider})
    with patch("fleet.dispatcher.ProviderPool.from_config", return_value=pool):
        return FleetRouter(config)


def _stub_provider(generate_return):
    provider = AsyncMock()
    provider.name = "ollama"
    provider.generate = AsyncMock(return_value=generate_return)
    return provider


# --------------------------------------------------------------------------- #
# WIRE 1 — classifier.mode == "llm"                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_mode_classifies_via_llm_provider():
    """mode='llm' + a provider returning a JSON tag → the router classifies
    through the LLM (provider is called, returned tag is honored)."""
    provider = _stub_provider(['{"tag": "math", "confidence": 0.95}'])
    config = _legacy(classifier=ClassifierConfig(mode="llm", llm_model="qwen-tiny"))
    router = _make_router(config, stub_provider=provider)

    assert isinstance(router._llm_classifier, LLMClassifier)

    with patch.object(router._registry, "get_best_for_tag", return_value="m") as best, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "42"}
        result = await router.ask("what is 6 times 7")

    assert result == "42"
    # The LLM provider actually did the classification...
    provider.generate.assert_awaited()
    # ...and its tag flowed into routing.
    best.assert_called_once_with("math")


@pytest.mark.asyncio
async def test_keyword_mode_never_builds_llm_classifier():
    """Default mode='keyword' → no LLM classifier built, keyword path used."""
    router = _make_router(_legacy(classifier=ClassifierConfig(mode="keyword")))
    assert router._llm_classifier is None

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)) as kw, \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "ok"}
        await router.ask("write a function")

    kw.assert_called_once_with("write a function")


@pytest.mark.asyncio
async def test_llm_mode_empty_model_falls_back_to_keyword(caplog):
    """mode='llm' but llm_model empty → degrade to keyword with a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        router = _make_router(_legacy(classifier=ClassifierConfig(mode="llm", llm_model="")))

    assert router._llm_classifier is None
    assert any("llm_model is empty" in r.message for r in caplog.records)

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)) as kw, \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "ok"}
        await router.ask("anything")

    kw.assert_called_once()


@pytest.mark.asyncio
async def test_llm_mode_missing_provider_degrades_to_keyword(caplog):
    """mode='llm' but the model's provider isn't in the pool → keyword + warn."""
    import logging

    config = _legacy(
        classifier=ClassifierConfig(mode="llm", llm_model="ghost"),
        models={"ghost": ModelEntry(provider="nonexistent")},
    )
    with caplog.at_level(logging.WARNING):
        router = _make_router(config)

    assert router._llm_classifier is None
    assert any("not in pool" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# WIRE 2 — retrieval.enabled                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retrieval_augments_dispatched_prompt_only():
    """Enabled + tag configured + non-empty context → the dispatcher receives
    the augmented prompt while classification saw the ORIGINAL."""
    config = _legacy(retrieval=RetrievalConfig(enabled=True, tags=["code"], provider="noop"))
    router = _make_router(config)

    stub = AsyncMock()
    stub.retrieve = AsyncMock(return_value="RETRIEVED CONTEXT:\nfacts")
    router._retrieval = stub

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)) as clf, \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "answer"}
        await router.ask("original prompt")

    # Classification saw the original, untouched prompt.
    clf.assert_called_once_with("original prompt")
    # Retrieval queried with the original prompt and the configured budget.
    stub.retrieve.assert_awaited_once_with("original prompt", 4000)
    # The dispatcher received the context-prepended prompt.
    assert dispatch.call_args[0][0] == "RETRIEVED CONTEXT:\nfacts\n\noriginal prompt"


@pytest.mark.asyncio
async def test_retrieval_skipped_when_tag_not_configured():
    """Tag not in retrieval.tags → prompt unchanged, retrieve never called."""
    config = _legacy(retrieval=RetrievalConfig(enabled=True, tags=["math"], provider="noop"))
    router = _make_router(config)

    stub = AsyncMock()
    stub.retrieve = AsyncMock(return_value="should not be used")
    router._retrieval = stub

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "answer"}
        await router.ask("original prompt")

    stub.retrieve.assert_not_awaited()
    assert dispatch.call_args[0][0] == "original prompt"


@pytest.mark.asyncio
async def test_retrieval_disabled_builds_nothing_and_leaves_prompt():
    """Disabled (the shipped default) → no provider built, prompt unchanged."""
    config = _legacy(retrieval=RetrievalConfig(enabled=False, tags=["code"]))
    router = _make_router(config)

    assert router._retrieval is None

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "answer"}
        await router.ask("original prompt")

    assert dispatch.call_args[0][0] == "original prompt"


@pytest.mark.asyncio
async def test_retrieval_empty_context_leaves_prompt_unchanged():
    """Empty context (e.g. NoOp / missing API key) → prompt unchanged."""
    config = _legacy(retrieval=RetrievalConfig(enabled=True, tags=["code"], provider="noop"))
    router = _make_router(config)

    stub = AsyncMock()
    stub.retrieve = AsyncMock(return_value="")
    router._retrieval = stub

    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="m"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"m": "answer"}
        await router.ask("original prompt")

    stub.retrieve.assert_awaited_once()
    assert dispatch.call_args[0][0] == "original prompt"


@pytest.mark.asyncio
async def test_retrieval_not_applied_on_force_model_path():
    """force_model fast path bypasses classification AND retrieval entirely."""
    config = _legacy(retrieval=RetrievalConfig(enabled=True, tags=["code"], provider="noop"))
    router = _make_router(config)

    stub = AsyncMock()
    stub.retrieve = AsyncMock(return_value="RETRIEVED CONTEXT:\nfacts")
    router._retrieval = stub

    with patch.object(router._classifier, "classify") as clf, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as dispatch:
        dispatch.return_value = {"forced": "answer"}
        result = await router.ask("original prompt", force_model="forced")

    assert result == "answer"
    stub.retrieve.assert_not_awaited()
    clf.assert_not_called()
    assert dispatch.call_args[0][0] == "original prompt"


# --------------------------------------------------------------------------- #
# Construction wiring (factory selection from config)                          #
# --------------------------------------------------------------------------- #


def test_retrieval_provider_constructed_from_config():
    websearch = _make_router(_legacy(retrieval=RetrievalConfig(enabled=True, provider="websearch")))
    assert isinstance(websearch._retrieval, WebSearchRetrieval)

    noop = _make_router(_legacy(retrieval=RetrievalConfig(enabled=True, provider="noop")))
    assert isinstance(noop._retrieval, NoOpRetrieval)

    off = _make_router(_legacy(retrieval=RetrievalConfig(enabled=False, provider="websearch")))
    assert off._retrieval is None
