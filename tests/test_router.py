import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fleet.router import (
    FleetRouter,
    ERROR_MODEL_FAILED,
    ERROR_ALL_MODELS_FAILED,
    ERROR_NO_MODEL,
    ERROR_NO_MODELS,
)
from fleet.config import Config, SamplingConfig, SynthesisConfig, ThresholdConfig


@pytest.fixture
def router():
    # Existing tests assert against the heuristic synthesizer's behavior on
    # the legacy single-call code path (`dispatcher.run`). The new max-quality
    # defaults — single_confidence>1 (always parallel) + samples>=3 per tag
    # (always verifier path via run_multi) — would route past that code path
    # entirely. Restore the legacy knobs here so these targeted tests keep
    # exercising the dispatcher.run heuristic branch. End-to-end max-quality
    # behavior is covered by test_router_verifier.py and test_config.py.
    config = Config(
        synthesis=SynthesisConfig(mode="heuristic"),
        thresholds=ThresholdConfig(single_confidence=0.8),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
    )
    return FleetRouter(config)


@pytest.mark.asyncio
async def test_single_mode(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="deepseek-v4-pro"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"deepseek-v4-pro": "def foo():\n    pass"}

        result = await router.ask("write a function")
        assert result == "def foo():\n    pass"
        mock_dispatch.assert_awaited_once()
        # Should only call 1 model
        assert len(mock_dispatch.call_args[0][1]) == 1


@pytest.mark.asyncio
async def test_parallel_mode(router):
    with patch.object(router._classifier, "classify", return_value=("creative", 0.6)), \
         patch.object(router._registry, "models_for_tag", return_value=["glm-5.1", "minimax-m2.7"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(router._synthesizer, "pick", return_value="best result"):
        mock_dispatch.return_value = {"glm-5.1": "a", "minimax-m2.7": "b"}

        result = await router.ask("write a story")
        assert result == "best result"
        # Should call 2 models
        assert len(mock_dispatch.call_args[0][1]) == 2


@pytest.mark.asyncio
async def test_force_model(router):
    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"gpt-4": "forced result"}

        result = await router.ask("prompt", force_model="gpt-4")
        assert result == "forced result"
        mock_dispatch.assert_awaited_once()
        assert mock_dispatch.call_args[0][1] == ["gpt-4"]


@pytest.mark.asyncio
async def test_force_parallel(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "models_for_tag", return_value=["model-a", "model-b"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(router._synthesizer, "pick", return_value="parallel result"):
        mock_dispatch.return_value = {"model-a": "a", "model-b": "b"}

        result = await router.ask("prompt", force_parallel=True)
        assert result == "parallel result"
        assert len(mock_dispatch.call_args[0][1]) == 2


@pytest.mark.asyncio
async def test_single_fallback_when_primary_fails(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="primary-model"), \
         patch.object(router._registry, "all_available", return_value=["fallback-model"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.side_effect = [
            {"primary-model": None},
            {"fallback-model": "fallback result"},
        ]

        result = await router.ask("prompt")
        assert result == "fallback result"
        assert mock_dispatch.await_count == 2


@pytest.mark.asyncio
async def test_no_models_available(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.6)), \
         patch.object(router._registry, "models_for_tag", return_value=[]), \
         patch.object(router._registry, "all_available", return_value=[]):
        result = await router.ask("prompt")
        assert result == ERROR_NO_MODELS


@pytest.mark.asyncio
async def test_system_prompt_forwarded(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": "result"}

        result = await router.ask("prompt", system="You are helpful")
        assert result == "result"
        assert mock_dispatch.call_args.kwargs["system"] == "You are helpful"


@pytest.mark.asyncio
async def test_system_prompt_from_config_when_caller_omits():
    """When the caller passes system=None, the config-level default is used."""
    cfg = Config(
        thresholds=ThresholdConfig(single_confidence=0.8),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        synthesis=SynthesisConfig(mode="heuristic"),
    )
    cfg.system_prompt = "You are a research agent."
    router = FleetRouter(cfg)
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": "result"}

        result = await router.ask("prompt")
        assert result == "result"
        assert mock_dispatch.call_args.kwargs["system"] == "You are a research agent."


@pytest.mark.asyncio
async def test_caller_system_overrides_config_default():
    """An explicit system= arg takes precedence over the config default."""
    cfg = Config(
        thresholds=ThresholdConfig(single_confidence=0.8),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
        synthesis=SynthesisConfig(mode="heuristic"),
    )
    cfg.system_prompt = "You are a research agent."
    router = FleetRouter(cfg)
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": "result"}

        result = await router.ask("prompt", system="Override prompt")
        assert result == "result"
        assert mock_dispatch.call_args.kwargs["system"] == "Override prompt"


@pytest.mark.asyncio
async def test_empty_string_response_not_treated_as_failure(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"model": ""}

        result = await router.ask("prompt")
        assert result == ""


@pytest.mark.asyncio
async def test_force_model_returns_error_when_model_fails(router):
    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"gpt-4": None}

        result = await router.ask("prompt", force_model="gpt-4")
        assert ERROR_MODEL_FAILED in result
        assert "gpt-4" in result  # error includes model name for context
        mock_dispatch.assert_awaited_once()
        assert mock_dispatch.call_args[0][1] == ["gpt-4"]


@pytest.mark.asyncio
async def test_force_model_all_thinking_does_not_leak_raw_cot(router):
    """Regression: a forced model whose entire output is a <think> block
    strips to "". `strip_thinking(result) or result` returned the RAW think
    block — the exact CoT leak this guards against. It must return the failure
    sentinel, never text containing '<think>'."""
    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"gpt-4": "<think>secret reasoning the user must never see</think>"}

        result = await router.ask("prompt", force_model="gpt-4")
        assert "<think>" not in result
        assert result == ERROR_ALL_MODELS_FAILED


@pytest.mark.asyncio
async def test_force_model_unclosed_thinking_does_not_leak_raw_cot(router):
    """Truncated-mid-thought variant: an unclosed <think> tag also strips to
    "" and must not leak."""
    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"gpt-4": "<think>reasoning that got cut off and never"}

        result = await router.ask("prompt", force_model="gpt-4")
        assert "<think>" not in result
        assert result == ERROR_ALL_MODELS_FAILED


@pytest.mark.asyncio
async def test_single_primary_all_thinking_does_not_leak_raw_cot(router):
    """Regression: the _single primary return used `strip_thinking(result) or
    result`, leaking a raw all-CoT primary output. It must surface the failure
    sentinel instead."""
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="primary-model"), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.return_value = {"primary-model": "<think>only internal reasoning here</think>"}

        result = await router.ask("prompt")
        assert "<think>" not in result
        assert result == ERROR_ALL_MODELS_FAILED


@pytest.mark.asyncio
async def test_single_fallback_skips_all_thinking_model(router):
    """Regression: the _single fallback loop used `strip_thinking(fb) or fb`,
    so an all-CoT fallback leaked raw reasoning. The loop must now skip a
    fallback that strips to "" and try the next model."""
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="primary-model"), \
         patch.object(router._registry, "all_available", return_value=["fb-cot", "fb-good"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.side_effect = [
            {"primary-model": None},
            {"fb-cot": "<think>all reasoning, no answer</think>", "fb-good": "real answer"},
        ]

        result = await router.ask("prompt")
        assert "<think>" not in result
        assert result == "real answer"


@pytest.mark.asyncio
async def test_single_no_model_for_tag(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value=None):
        result = await router.ask("prompt")
        assert result == f"{ERROR_NO_MODEL} for tag: code"


@pytest.mark.asyncio
async def test_single_both_primary_and_fallback_fail(router):
    with patch.object(router._classifier, "classify", return_value=("code", 0.95)), \
         patch.object(router._registry, "get_best_for_tag", return_value="primary-model"), \
         patch.object(router._registry, "all_available", return_value=["fallback-model"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.side_effect = [
            {"primary-model": None},
            {"fallback-model": None},
        ]

        result = await router.ask("prompt")
        assert result == ERROR_ALL_MODELS_FAILED
        assert mock_dispatch.await_count == 2


@pytest.mark.asyncio
async def test_parallel_falls_back_to_all_available(router):
    with patch.object(router._classifier, "classify", return_value=("creative", 0.6)), \
         patch.object(router._registry, "models_for_tag", return_value=[]), \
         patch.object(router._registry, "all_available", return_value=["glm-5.1"]), \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(router._synthesizer, "pick", return_value="fallback result"):
        mock_dispatch.return_value = {"glm-5.1": "fallback result"}

        result = await router.ask("prompt")
        assert result == "fallback result"
        mock_dispatch.assert_awaited_once()
        assert mock_dispatch.call_args[0][1] == ["glm-5.1"]
