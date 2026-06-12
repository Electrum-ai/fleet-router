"""Tests for scripts/fleet-ensure-proxy.py security helpers.

The script lives outside the package and has a hyphenated filename, so we
load it via importlib from its path. We test the importable helpers
(private run-dir creation + the 'is this our process?' kill guard); the
end-to-end boot flow spawns a real subprocess and is exercised manually.
"""
from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fleet-ensure-proxy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fleet_ensure_proxy", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def test_ensure_run_dir_creates_0700(mod, tmp_path):
    target = tmp_path / "nested" / "run"
    returned = mod.ensure_run_dir(target)
    assert returned == target
    assert target.is_dir()
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o700


def test_ensure_run_dir_tightens_preexisting_loose_dir(mod, tmp_path):
    target = tmp_path / "run"
    target.mkdir(mode=0o777)
    os.chmod(target, 0o777)  # simulate a world-writable pre-existing dir
    mod.ensure_run_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_is_our_fleet_proxy_rejects_pid_1(mod):
    # init/launchd — alive and not ours to kill.
    assert mod.is_our_fleet_proxy(1) is False


def test_is_our_fleet_proxy_rejects_dead_pid(mod):
    # A pid that does not exist must never be considered killable.
    dead = 2_000_000_000
    assert mod.is_our_fleet_proxy(dead) is False


def test_is_our_fleet_proxy_rejects_foreign_command(mod, monkeypatch):
    """A live, us-owned process that is NOT the fleet proxy must be refused —
    this is the arbitrary-process-kill guard."""
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: None)  # pretend alive
    monkeypatch.setattr(mod, "_process_uid", lambda pid: os.getuid())
    monkeypatch.setattr(mod, "_process_command", lambda pid: "/usr/bin/vim notes.txt")
    assert mod.is_our_fleet_proxy(4242) is False


def test_is_our_fleet_proxy_accepts_our_proxy(mod, monkeypatch):
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(mod, "_process_uid", lambda pid: os.getuid())
    monkeypatch.setattr(
        mod, "_process_command",
        lambda pid: "/x/venv/bin/fleet --serve --port 8765 --api-key fleet-local",
    )
    assert mod.is_our_fleet_proxy(4242) is True


def test_is_our_fleet_proxy_rejects_foreign_uid(mod, monkeypatch):
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(mod, "_process_uid", lambda pid: os.getuid() + 9999)
    monkeypatch.setattr(
        mod, "_process_command",
        lambda pid: "/x/venv/bin/fleet --serve --port 8765",
    )
    assert mod.is_our_fleet_proxy(4242) is False


def test_runtime_paths_under_private_dir(mod):
    """Pidfile/logfile/lockfile must live under ~/.fleet/run, never $TMPDIR."""
    assert mod.RUN_DIR == Path.home() / ".fleet" / "run"
    for p in (mod.PIDFILE, mod.LOGFILE, mod.LOCKFILE):
        assert p.parent == mod.RUN_DIR


def test_write_pidfile_refuses_symlink(mod, tmp_path, monkeypatch):
    """O_NOFOLLOW: a pre-planted symlink at the pidfile path must not be
    followed — the write fails instead of clobbering the symlink target."""
    real_target = tmp_path / "victim"
    real_target.write_text("important\n")
    link = tmp_path / "fleet-proxy.pid"
    link.symlink_to(real_target)
    monkeypatch.setattr(mod, "PIDFILE", link)
    with pytest.raises(OSError):
        mod.write_pidfile(1234)
    # Victim untouched.
    assert real_target.read_text() == "important\n"
