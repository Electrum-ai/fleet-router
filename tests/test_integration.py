import pytest
import asyncio
from fleet.router import FleetRouter
from fleet.config import Config, ModelEntry

@pytest.mark.asyncio
async def test_end_to_end_mocked():
    """Smoke test the full pipeline with mocked Ollama."""
    from unittest.mock import patch, AsyncMock

    config = Config(models={"deepseek-v4-pro": ModelEntry(tags=["code"], priority=1)})
    router = FleetRouter(config)
    router._registry._available = {"deepseek-v4-pro"}

    with patch.object(router._dispatcher, "run", new_callable=AsyncMock) as mock_disp:
        mock_disp.return_value = {"deepseek-v4-pro": "def foo(): pass"}
        result = await router.ask("write a python function")
        assert "def foo" in str(result)
