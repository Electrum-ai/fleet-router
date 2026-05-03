import pytest
from unittest.mock import AsyncMock, patch

from fleet.dispatcher import EnsembleDispatcher
from fleet.config import Config


@pytest.mark.asyncio
async def test_dispatch_single():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"response": "hello"})
        mock_response.raise_for_status = AsyncMock()
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": "hello"}


@pytest.mark.asyncio
async def test_dispatch_parallel():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        async def mock_json():
            return {"response": "result"}

        mock_response = AsyncMock()
        mock_response.json = mock_json
        mock_response.raise_for_status = AsyncMock()
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await disp.run("hi", ["glm-5.1", "minimax-m2.7"])
        assert set(result.keys()) == {"glm-5.1", "minimax-m2.7"}
