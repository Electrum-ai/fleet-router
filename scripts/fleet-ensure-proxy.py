#!/usr/bin/env python3
"""Idempotent boot for the Fleet Router proxy.

Designed to be invoked from a Claude Code SessionStart hook. Concurrent
SessionStart hooks (e.g. two chats opening at once) coordinate via flock
so only one process actually launches `fleet --serve`.

Exit codes:
    0  proxy is healthy
    1  proxy failed to come up within deadline
    2  fleet binary not found

The proxy itself is detached via os.setsid so it survives this script
exiting and the hook closing its stdio.

Security notes
--------------
Runtime state (pidfile, logfile, lockfile) lives under a PRIVATE
``~/.fleet/run`` directory created mode 0700 — not world-writable
``$TMPDIR``/``/tmp``, where a predictable, attacker-pre-creatable path
enabled symlink and pid-confusion attacks. The logfile and pidfile are
opened ``O_NOFOLLOW`` so a pre-planted symlink can't redirect the write,
and we refuse to signal any pid from the pidfile unless it is alive, owned
by us, AND recognizably the fleet proxy.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path("/Users/bistrocloud/fleet-router")
VENV_FLEET = REPO_ROOT / "venv" / "bin" / "fleet"

PORT = int(os.environ.get("FLEET_PORT", "8765"))
API_KEY = os.environ.get("FLEET_API_KEY", "fleet-local")

# Private runtime dir — NOT world-writable $TMPDIR. Created 0700 in
# ensure_run_dir() before any file under it is opened.
RUN_DIR = Path.home() / ".fleet" / "run"
PIDFILE = RUN_DIR / "fleet-proxy.pid"
LOGFILE = RUN_DIR / "fleet-proxy.log"
LOCKFILE = RUN_DIR / "fleet-ensure-proxy.lock"

HEALTH_URL = f"http://127.0.0.1:{PORT}/healthz"
# Cold start can include sentence-transformers download/load; give it room.
BOOT_DEADLINE_S = int(os.environ.get("FLEET_BOOT_DEADLINE_S", "60"))


def log(msg: str) -> None:
    print(f"[fleet-ensure-proxy] {msg}", file=sys.stderr)


def ensure_run_dir(path: Path | None = None) -> Path:
    """Create the private runtime dir 0700 and tighten its mode if it
    pre-existed with looser permissions. Returns the dir."""
    target = path if path is not None else RUN_DIR
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except OSError:
        pass
    return target


def is_pid_alive(pid: int) -> bool:
    """True only when the process exists AND is signalable by us. On EPERM
    the process exists but is owned by another user — for the kill-decision
    path that means 'not ours', so we deliberately return False to keep the
    kill window as narrow as possible."""
    try:
        os.kill(pid, 0)
    except OSError:
        # ESRCH → gone; EPERM → exists but owned by another user. Either way,
        # for the kill-decision path it is not a process we may signal.
        return False
    return True


def _process_uid(pid: int) -> int | None:
    """Owning uid of a pid, via procfs where available, else `ps`."""
    proc_status = Path(f"/proc/{pid}")
    if proc_status.exists():
        try:
            return proc_status.stat().st_uid
        except OSError:
            return None
    try:
        out = subprocess.run(
            ["ps", "-o", "uid=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    val = out.stdout.strip()
    try:
        return int(val) if val else None
    except ValueError:
        return None


def _process_command(pid: int) -> str:
    """Command line of a pid via `ps` (portable across macOS + Linux)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def is_our_fleet_proxy(pid: int) -> bool:
    """Refuse to treat a pid as killable unless it is (a) a real, signalable
    process, (b) owned by the current uid, and (c) recognizably the fleet
    proxy. Guards against SIGKILLing an arbitrary process whose pid happened
    to be written into (or planted in) the pidfile."""
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        # ESRCH (gone) or EPERM (another user's process) — never ours.
        return False
    uid = _process_uid(pid)
    if uid is None or uid != os.getuid():
        return False
    cmd = _process_command(pid)
    return "fleet" in cmd and "--serve" in cmd


def read_pidfile() -> int | None:
    try:
        return int(PIDFILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_pidfile(pid: int) -> None:
    """Write the pidfile O_NOFOLLOW so a pre-planted symlink can't redirect
    the write to a file we don't own."""
    fd = os.open(
        PIDFILE,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(fd, f"{pid}\n".encode("utf-8"))
    finally:
        os.close(fd)


def open_logfile():
    """Open the logfile O_NOFOLLOW|O_APPEND, returning a binary file object
    suitable for a subprocess's stdout/stderr."""
    fd = os.open(
        LOGFILE,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
        0o600,
    )
    return os.fdopen(fd, "ab", buffering=0)


def healthz_ok(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("ok"))
    except (urllib.error.URLError, socket.timeout, OSError, ValueError):
        return False


def port_in_use() -> bool:
    """Detects an unrelated process holding the port — distinct from a
    healthy fleet proxy responding on /healthz."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", PORT))
            return True
        except OSError:
            return False


def already_running() -> bool:
    pid = read_pidfile()
    if pid is None or not is_pid_alive(pid):
        return False
    return healthz_ok()


def spawn_proxy() -> int:
    if not VENV_FLEET.exists():
        log(f"fleet binary not found at {VENV_FLEET}")
        sys.exit(2)

    log_fh = open_logfile()
    proc = subprocess.Popen(
        [str(VENV_FLEET), "--serve", "--port", str(PORT), "--api-key", API_KEY],
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        # Detach so the proxy outlives this script and the hook's stdio.
        start_new_session=True,
        close_fds=True,
    )
    write_pidfile(proc.pid)
    return proc.pid


def wait_for_health(deadline_s: int) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if healthz_ok():
            return True
        time.sleep(1.0)
    return False


def main() -> int:
    ensure_run_dir()

    if already_running():
        log(f"proxy already healthy on port {PORT}")
        return 0

    LOCKFILE.touch(exist_ok=True)
    with open(LOCKFILE, "r+") as lock_fh:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another SessionStart is already booting the proxy — wait for it.
            log("another instance is starting the proxy; waiting for health")
            if wait_for_health(BOOT_DEADLINE_S):
                return 0
            log(f"proxy did not become healthy within {BOOT_DEADLINE_S}s")
            return 1

        # We hold the exclusive lock — re-check under the lock and start.
        if already_running():
            log("proxy became healthy while acquiring lock")
            return 0

        # Stale pidfile or unrelated port holder?
        pid = read_pidfile()
        if pid is not None and is_our_fleet_proxy(pid):
            # Our proxy is alive but /healthz is not OK — kill and respawn.
            log(f"pid {pid} is our fleet proxy but unhealthy; terminating")
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(20):
                    if not is_pid_alive(pid):
                        break
                    time.sleep(0.25)
                if is_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            except OSError as exc:
                log(f"could not terminate pid {pid}: {exc}")
        elif pid is not None:
            # A pid is recorded but it isn't our proxy (foreign owner, wrong
            # command, or recycled pid). Never signal it.
            log(f"pidfile pid {pid} is not our fleet proxy; refusing to kill")

        if port_in_use():
            log(f"port {PORT} held by an unknown process; not auto-killing")
            return 1

        new_pid = spawn_proxy()
        log(f"spawned fleet --serve (pid {new_pid}); polling {HEALTH_URL}")

    if wait_for_health(BOOT_DEADLINE_S):
        log("proxy is healthy")
        return 0
    log(f"proxy did not become healthy within {BOOT_DEADLINE_S}s — see {LOGFILE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
