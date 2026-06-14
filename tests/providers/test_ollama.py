import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from fleet.providers.base import GenerateRequest
from fleet.providers.ollama import OllamaProvider


def _setup_post(mock_post, response_json=None, raise_for_status_error=None, post_error=None):
    if post_error:
        mock_post.side_effect = post_error
        return
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=response_json or {"response": "ok"})
    if raise_for_status_error:
        mock_response.raise_for_status = MagicMock(side_effect=raise_for_status_error)
    else:
        mock_response.raise_for_status = MagicMock()
    mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.return_value.__aexit__ = AsyncMock(return_value=False)


@pytest.mark.asyncio
async def test_ollama_single_sample():
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi")
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post, response_json={"response": "answer"})
        result = await p.generate(req)
    await p.aclose()
    assert result == ["answer"]


@pytest.mark.asyncio
async def test_ollama_multi_sample():
    """samples=N must produce N independent calls and return N entries."""
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi", samples=3)
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post, response_json={"response": "x"})
        result = await p.generate(req)
    await p.aclose()
    assert len(result) == 3
    assert all(r == "x" for r in result)
    assert mock_post.call_count == 3


@pytest.mark.asyncio
async def test_ollama_passes_temperature():
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi", temperature=0.9)
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post)
        await p.generate(req)
    await p.aclose()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_ollama_http_error_yields_none():
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi")
    err = aiohttp.ClientResponseError(
        request_info=AsyncMock(real_url="http://x"), history=(), status=500
    )
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post, raise_for_status_error=err)
        result = await p.generate(req)
    await p.aclose()
    assert result == [None]


@pytest.mark.asyncio
async def test_ollama_zero_samples_returns_empty():
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi", samples=0)
    result = await p.generate(req)
    await p.aclose()
    assert result == []


@pytest.mark.asyncio
async def test_ollama_session_reused_across_calls():
    """Backpressure regression guard: the v1 provider opened a fresh session
    per generate(). The new code reuses one session — verify that's true."""
    p = OllamaProvider()
    req = GenerateRequest(model="glm", prompt="hi")
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post, response_json={"response": "ok"})
        await p.generate(req)
        first_session = p._session
        await p.generate(req)
        assert p._session is first_session
    await p.aclose()
    assert p._session is None  # closed


@pytest.mark.asyncio
async def test_ollama_semaphore_caps_concurrent_requests():
    """With max_concurrent=2 and 5 samples, only 2 should be in flight at
    any moment. The mock's __aenter__ holds briefly so peers pile up;
    we measure peak concurrency observed inside __aenter__."""
    p = OllamaProvider(max_concurrent=2)
    req = GenerateRequest(model="glm", prompt="hi", samples=5)

    in_flight = 0
    peak = 0

    async def aenter(_self=None):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"response": "ok"})
        mock_response.raise_for_status = MagicMock()
        return mock_response

    def post_factory(*args, **kwargs):
        cm = AsyncMock()
        cm.__aenter__ = aenter
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    with patch("aiohttp.ClientSession.post", side_effect=post_factory):
        await p.generate(req)
    await p.aclose()
    assert peak <= 2, f"semaphore should cap to 2 in-flight, saw {peak}"
    assert peak >= 1  # sanity: at least one was in flight


@pytest.mark.asyncio
async def test_ollama_per_request_timeout_overrides_session_default():
    """A GenerateRequest.timeout must be passed to session.post as a
    ClientTimeout, overriding the session-wide default per request."""
    p = OllamaProvider(timeout=60)
    req = GenerateRequest(model="glm", prompt="hi", timeout=240)
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post)
        await p.generate(req)
    await p.aclose()
    passed = mock_post.call_args.kwargs["timeout"]
    assert isinstance(passed, aiohttp.ClientTimeout)
    assert passed.total == 240


@pytest.mark.asyncio
async def test_ollama_no_per_request_timeout_falls_back_to_session():
    """Without request.timeout, no per-request timeout kwarg is sent — the
    shared session default applies."""
    p = OllamaProvider(timeout=60)
    req = GenerateRequest(model="glm", prompt="hi")  # timeout left None
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post)
        await p.generate(req)
    await p.aclose()
    assert "timeout" not in mock_post.call_args.kwargs


@pytest.mark.asyncio
async def test_ollama_list_models():
    p = OllamaProvider()
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "models": [{"name": "glm-5.1:fp16"}, {"name": "deepseek:latest"}]
        })
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=False)
        models = await p.list_models()
    names = sorted(m.name for m in models)
    assert names == ["deepseek", "glm-5.1"]
    assert all(m.provider == "ollama" for m in models)


@pytest.mark.asyncio
async def test_ollama_list_models_handles_errors():
    p = OllamaProvider()
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = aiohttp.ClientError("down")
        models = await p.list_models()
    assert models == []


def test_ollama_headers_with_api_key():
    p = OllamaProvider(api_key="sk-secret")
    assert p._headers() == {
        "Accept": "application/json",
        "Authorization": "Bearer sk-secret",
    }


def test_ollama_headers_without_api_key():
    p = OllamaProvider(api_key="")
    assert p._headers() == {"Accept": "application/json"}


@pytest.mark.asyncio
async def test_ollama_generate_works_with_api_key():
    """Provider with api_key must still successfully dispatch."""
    p = OllamaProvider(api_key="sk-secret")
    req = GenerateRequest(model="glm", prompt="hi")
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_post(mock_post, response_json={"response": "ok"})
        result = await p.generate(req)
    await p.aclose()
    assert result == ["ok"]


@pytest.mark.asyncio
async def test_ollama_list_models_works_with_api_key():
    p = OllamaProvider(api_key="sk-secret")
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "models": [{"name": "glm-5.1:fp16"}, {"name": "deepseek:latest"}]
        })
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=False)
        models = await p.list_models()
    names = sorted(m.name for m in models)
    assert names == ["deepseek", "glm-5.1"]
