import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fleet.router import FleetRouter
from fleet.config import Config


@pytest.fixture
def router():
    config = Config()
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
