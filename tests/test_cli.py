"""Tests for the `fleet` CLI entrypoint.

When mocking FleetRouter, remember the CLI awaits BOTH `ask` and `aclose`
(the latter in a finally block to close the aiohttp pool). Tests must
configure both as AsyncMock or every test returns rc=1 from a TypeError
raised on `await MagicMock()`."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from fleet import cli


def test_build_parser_basic():
    parser = cli._build_parser()
    args = parser.parse_args(["hello", "world"])
    assert args.prompt == ["hello", "world"]
    assert args.parallel is False
    assert args.model is None
    assert args.config is None
    assert args.verbose is False


def test_build_parser_flags():
    parser = cli._build_parser()
    args = parser.parse_args([
        "--parallel", "--model", "glm-5.1", "--config", "/tmp/x.yaml",
        "-v", "do", "the", "thing",
    ])
    assert args.parallel is True
    assert args.model == "glm-5.1"
    assert args.config == "/tmp/x.yaml"
    assert args.verbose is True
    assert args.prompt == ["do", "the", "thing"]


def test_main_prints_string_result(capsys):
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(return_value="the answer")
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["hello"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "the answer"


def test_main_prints_dict_result_with_headers(capsys):
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(return_value={
            "glm-5.1": "answer A",
            "minimax-m2.7": "answer B",
        })
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["compare", "models"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "--- glm-5.1 ---" in captured.out
    assert "answer A" in captured.out
    assert "--- minimax-m2.7 ---" in captured.out
    assert "answer B" in captured.out


def test_main_passes_force_model_normalized(capsys):
    """--model glm-5.1:cloud must be normalized to glm-5.1 before dispatch."""
    with patch("fleet.cli.FleetRouter") as router_cls:
        ask = AsyncMock(return_value="ok")
        router_cls.return_value.ask = ask
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--model", "glm-5.1:cloud", "hi"])
    assert rc == 0
    ask.assert_awaited_once()
    assert ask.call_args.kwargs["force_model"] == "glm-5.1"


def test_main_passes_force_parallel(capsys):
    with patch("fleet.cli.FleetRouter") as router_cls:
        ask = AsyncMock(return_value="ok")
        router_cls.return_value.ask = ask
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--parallel", "hello"])
    assert rc == 0
    assert ask.call_args.kwargs["force_parallel"] is True


def test_main_returns_exit_code_2_on_sentinel_error(capsys):
    """Sentinel error strings (start with '(') signal routing failure to shells."""
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(
            return_value="(no model available) for tag: code"
        )
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["hello"])
    assert rc == 2


def test_main_returns_exit_code_1_on_unexpected_exception(capsys):
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(side_effect=RuntimeError("boom"))
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["hello"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "RuntimeError" in captured.err
    assert "boom" in captured.err


def test_main_no_argv_uses_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["fleet", "from", "argv"])
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(return_value="ok")
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main()
    assert rc == 0


def test_main_missing_prompt_returns_error(capsys):
    """Without `prompt` and without `--eval`, the CLI prints an error and
    exits 1 (argparse no longer rejects empty args because --eval can fill
    that slot)."""
    rc = cli.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing prompt" in captured.err


def test_eval_command_runs_and_returns_aggregates(tmp_path, capsys):
    """`fleet --eval <dir>` runs the eval harness and prints aggregates."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "math.jsonl").write_text(
        '{"prompt": "1+1?", "tag": "math", "expected": 2}\n'
    )
    with patch("fleet.cli.FleetRouter") as router_cls:
        router_cls.return_value.ask = AsyncMock(return_value="the answer is 2")
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--eval", str(fixtures)])
    captured = capsys.readouterr()
    assert rc == 0
    output = json.loads(captured.out)
    assert output["n_cases"] == 1
    assert output["aggregates"]["math"]["pass_rate"] == 1.0


def test_eval_command_missing_fixtures_dir(tmp_path, capsys):
    rc = cli.main(["--eval", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "fixtures directory not found" in captured.err


# ---------- H1: refuse non-loopback bind without an API key ----------


def test_serve_non_loopback_without_key_refuses(capsys):
    """`fleet --serve --host 0.0.0.0` with no key must error out WITHOUT
    binding — otherwise it's an open relay to the operator's Ollama."""
    with patch("fleet.cli.FleetRouter") as router_cls, \
            patch("fleet.proxy.serve") as serve:
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--serve", "--host", "0.0.0.0"])
    captured = capsys.readouterr()
    assert rc == 1
    serve.assert_not_called()
    assert "refusing to bind" in captured.err
    assert "--api-key" in captured.err


def test_serve_non_loopback_with_key_proceeds():
    with patch("fleet.cli.FleetRouter") as router_cls, \
            patch("fleet.proxy.serve") as serve:
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--serve", "--host", "0.0.0.0", "--api-key", "secret"])
    assert rc == 0
    serve.assert_called_once()
    assert serve.call_args.kwargs["host"] == "0.0.0.0"
    assert serve.call_args.kwargs["api_key"] == "secret"


def test_serve_loopback_without_key_proceeds():
    """The single-user local default: 127.0.0.1 with no key stays allowed."""
    with patch("fleet.cli.FleetRouter") as router_cls, \
            patch("fleet.proxy.serve") as serve:
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--serve", "--host", "127.0.0.1"])
    assert rc == 0
    serve.assert_called_once()


def test_serve_localhost_without_key_proceeds():
    with patch("fleet.cli.FleetRouter") as router_cls, \
            patch("fleet.proxy.serve") as serve:
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--serve", "--host", "localhost"])
    assert rc == 0
    serve.assert_called_once()


def test_serve_lan_ip_without_key_refuses(capsys):
    with patch("fleet.cli.FleetRouter") as router_cls, \
            patch("fleet.proxy.serve") as serve:
        router_cls.return_value.aclose = AsyncMock()
        rc = cli.main(["--serve", "--host", "192.168.1.50"])
    assert rc == 1
    serve.assert_not_called()
