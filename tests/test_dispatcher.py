import asyncio

import aiohttp
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fleet.dispatcher import EnsembleDispatcher
from fleet.config import Config, ModelEntry, ThresholdConfig, load_config


def _setup_mock_post(
    mock_post,
    response_json=None,
    raise_for_status_error=None,
    post_error=None,
):
    """Configure mock aiohttp.ClientSession.post with desired behavior.

    Note: aiohttp's ClientResponse.raise_for_status is *synchronous*, so the
    mock must be a regular MagicMock — not an AsyncMock — or `side_effect`
    will not raise inline."""
    if post_error:
        mock_post.side_effect = post_error
        return

    mock_response = AsyncMock()
    mock_response.json = AsyncMock(
        return_value={"response": "hello"} if response_json is None else response_json
    )
    if raise_for_status_error:
        mock_response.raise_for_status = MagicMock(side_effect=raise_for_status_error)
    else:
        mock_response.raise_for_status = MagicMock()
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
async def test_dispatch_non_string_response_treated_as_failure():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={"response": 12345})
        result = await disp.run("hi", ["glm-5.1"])
        assert result == {"glm-5.1": None}


@pytest.mark.asyncio
async def test_dispatch_oversized_response_truncated():
    config = Config()
    disp = EnsembleDispatcher(config)

    huge = "x" * (4 * 1024 * 1024 + 100)
    with patch("aiohttp.ClientSession.post") as mock_post:
        _setup_mock_post(mock_post, response_json={"response": huge})
        result = await disp.run("hi", ["glm-5.1"])
        assert result["glm-5.1"] is not None
        assert len(result["glm-5.1"]) == 4 * 1024 * 1024


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


# ---------- CHANGE 1: per-class timeout budgets ----------


@pytest.mark.asyncio
async def test_run_multi_assigns_per_class_timeout_budgets():
    """Each model's GenerateRequest carries the budget for its model_class:
    reasoning gets the larger budget, chat the smaller, and an unconfigured
    model defaults to the chat budget."""
    config = Config(
        models={
            "reasoner": ModelEntry(tags=["math"], model_class="reasoning"),
            "chatter": ModelEntry(tags=["math"], model_class="chat"),
        },
        thresholds=ThresholdConfig(timeouts={"chat": 60, "reasoning": 240}),
    )
    disp = EnsembleDispatcher(config)

    captured: dict[str, float | None] = {}

    async def fake_generate(req):
        captured[req.model] = req.timeout
        return ["ok"]

    with patch.object(disp._default_provider, "generate", side_effect=fake_generate):
        await disp.run_multi("p", ["reasoner", "chatter", "unknown"], samples=1)

    assert captured["reasoner"] == 240
    assert captured["chatter"] == 60
    assert captured["unknown"] == 60  # unconfigured → chat budget


@pytest.mark.asyncio
async def test_run_multi_reasoning_budget_derived_from_legacy_parallel_timeout(tmp_path):
    """A config that predates `timeouts` and only sets `parallel_timeout` must
    still give a reasoning-class model the larger derived budget (>= 240) —
    not cut it off at the legacy chat budget."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "thresholds:\n"
        "  parallel_timeout: 45\n"
        "models:\n"
        "  thinker:\n"
        "    tags: [math]\n"
        "    class: reasoning\n"
        "  talker:\n"
        "    tags: [math]\n"
        "    class: chat\n"
    )
    config = load_config(config_path)
    disp = EnsembleDispatcher(config)

    captured: dict[str, float | None] = {}

    async def fake_generate(req):
        captured[req.model] = req.timeout
        return ["ok"]

    with patch.object(disp._default_provider, "generate", side_effect=fake_generate):
        await disp.run_multi("p", ["thinker", "talker"], samples=1)

    assert captured["thinker"] == 240  # derived max(45, 240)
    assert captured["talker"] == 45    # derived chat budget


@pytest.mark.asyncio
async def test_dispatcher_session_sized_to_chat_budget():
    """The shared session default is the CHAT budget — NOT the max. aiohttp's
    per-request ClientTimeout fully replaces (not caps) the session value, so a
    reasoning per-request override still gets its full 240s even though the
    session default is the smaller chat budget. Sizing the session to chat
    keeps no-per-request-timeout callers (judge/classifier) on the 60s default
    instead of inheriting a bloated max budget."""
    config = Config(
        thresholds=ThresholdConfig(timeouts={"chat": 60, "reasoning": 300}),
    )
    disp = EnsembleDispatcher(config)
    # Session default is the chat budget, not max(class budgets).
    assert disp._timeout == 60

    # End-to-end guarantee: a reasoning model still earns its full per-request
    # budget (300 here), which replaces the 60s session default.
    config2 = Config(
        models={"reasoner": ModelEntry(tags=["math"], model_class="reasoning")},
        thresholds=ThresholdConfig(timeouts={"chat": 60, "reasoning": 240}),
    )
    disp2 = EnsembleDispatcher(config2)
    captured: dict[str, float | None] = {}

    async def fake_generate(req):
        captured[req.model] = req.timeout
        return ["ok"]

    with patch.object(disp2._default_provider, "generate", side_effect=fake_generate):
        await disp2.run_multi("p", ["reasoner"], samples=1)
    # Per-request override (240) > session default (60): per-request replaces.
    assert captured["reasoner"] == 240
    assert disp2._timeout == 60


@pytest.mark.asyncio
async def test_dispatch_mixed_success_failure():
    config = Config()
    disp = EnsembleDispatcher(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        success_resp = AsyncMock()
        success_resp.json = AsyncMock(return_value={"response": "ok"})
        success_resp.raise_for_status = MagicMock()

        error_resp = AsyncMock()
        error_resp.raise_for_status = MagicMock(
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
