import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from fleet.router import FleetRouter
from fleet.config import Config, ModelEntry, SamplingConfig, SynthesisConfig


@pytest.mark.asyncio
async def test_end_to_end_heuristic_mode():
    """Smoke test the legacy heuristic synthesis path end-to-end. Heuristic
    mode + samples=1 is the only branch that still calls dispatcher.run
    (everything else now goes through run_multi for the verifier path)."""
    config = Config(
        models={"deepseek-v4-pro": ModelEntry(tags=["code"], priority=1)},
        synthesis=SynthesisConfig(mode="heuristic"),
        sampling=SamplingConfig(samples_by_tag={"default": 1}),
    )
    router = FleetRouter(config)
    router._registry._available = {"deepseek-v4-pro"}
    router._registry._refreshed = True

    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_disp:
        mock_disp.return_value = {"deepseek-v4-pro": "def foo(): pass"}
        result = await router.ask("write a python function")
        assert "def foo" in str(result)


@pytest.mark.asyncio
async def test_end_to_end_verifier_mode():
    """Smoke test the verifier synthesis path with multi-sample dispatch."""
    config = Config(
        models={"deepseek-v4-pro": ModelEntry(tags=["code"], priority=1)},
        synthesis=SynthesisConfig(mode="verifier", abstention_threshold=0.0),
        sampling=SamplingConfig(samples_by_tag={"code": 2, "default": 1}),
    )
    router = FleetRouter(config)
    router._registry._available = {"deepseek-v4-pro"}
    router._registry._refreshed = True

    with patch.object(router._dispatcher, "run_multi", new_callable=AsyncMock) as mock_multi, \
         patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_run:
        mock_multi.return_value = {
            "deepseek-v4-pro": ["def foo():\n    return 1", "def foo(): return 2"]
        }
        # Single-mode path also possible; mock both.
        mock_run.return_value = {"deepseek-v4-pro": "def foo():\n    return 1"}
        result = await router.ask("write a python function")
        assert "def foo" in str(result)
