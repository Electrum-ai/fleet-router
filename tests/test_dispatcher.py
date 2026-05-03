import asyncio

import aiohttp
import pytest
from unittest.mock import AsyncMock, patch

from fleet.dispatcher import EnsembleDispatcher
from fleet.config import Config


def _setup_mock_post(
    mock_post,
    response_json=None,
    raise_for_status_error=None,
    post_error=None,
):
    """Configure mock aiohttp.ClientSession.post with desired behavior."""
    if post_error:
        mock_post.side_effect = post_error
        return

    mock_response = AsyncMock()
    mock_response.json = AsyncMock(
        return_value={"response": "hello"} if response_json is None else response_json
    )
    if raise_for_status_error:
        mock_response.raise_for_status = AsyncMock(side_effect=raise_for_status_error)
    else:
        mock_response.raise_for_status = AsyncMock()
    mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.return_value.__aexit__ = AsyncMock(return_value=False)


@pytest.mark.asyncio
async def test_dispatch_single():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={"response": "hello"})
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": "hello"}


@pytest.mark.asyncio
async def test_dispatch_parallel():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={"response": "result"})
        result = await disp.run("hi", ["glm-5.1", "minimax-m2.7"])
        assert result == {"glm-5.1": "result", "minimax-m2.7": "result"}


@pytest.mark.asyncio
async def test_dispatch_http_error():
    config = Config()
    disp = EnsembleDispatcher(config)

    error = aiohttp.ClientResponseError(
        request_info=AsyncMock(real_url="http://localhost:11434/api/generate"),
        history=(),
        status=500,
    )
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, raise_for_status_error=error)
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": None}


@pytest.mark.asyncio
async def test_dispatch_timeout():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, post_error=asyncio.TimeoutError())
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": None}


@pytest.mark.asyncio
async def test_dispatch_missing_response_key():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={})
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": None}


@pytest.mark.asyncio
async def test_dispatch_exception_in_call():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, post_error=RuntimeError("boom"))
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": None}


@pytest.mark.asyncio
async def test_dispatch_empty_models():
    config = Config()
    disp = EnsembleDispatcher(config)

    result = await disp.run("hi", [])
    assert result == {}


@pytest.mark.asyncio
async def test_dispatch_system_prompt():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={"response": "system result"})
        result = await disp.run("hi", ["glm-5.1"], system="Be helpful")
        assert result == {"glm-5.1": "system result"}
        assert mock_post.call_args.kwargs["json"]["system"] == "Be helpful"


@pytest.mark.asyncio
async def test_dispatch_mixed_success_failure():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        success_resp = AsyncMock()
        success_resp.json = AsyncMock(return_value={"response": "ok"})
        success_resp.raise_for_status = AsyncMock()

        error_resp = AsyncMock()
        error_resp.raise_for_status = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=AsyncMock(real_url="http://localhost:11434/api/generate"),
                history=(),
                status=500,
            )
        )

        mock_post.return_value.__aenter__ = AsyncMock(
            side_effect=[success_resp, error_resp]
        )
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await disp.run("hi", ["glm-5.1", "minimax-m2.7"])
        assert result == {"glm-5.1": "ok", "minimax-m2.7": None}
