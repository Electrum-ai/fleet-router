# Fleet Router toggle for Claude Code.
# Source this from ~/.zshrc or ~/.bashrc:
#
#     source /path/to/fleet-router/scripts/fleet-toggle.sh
#
# Then in any shell:
#     fleet-on      → boot proxy, route `claude` through fleet → Ollama
#     fleet-off     → stop proxy, send `claude` back to Anthropic
#     fleet-status  → show current state
#
# Env vars are scoped to the shell that ran fleet-on; the proxy itself
# runs in the background and is shared across shells.

export FLEET_PORT="${FLEET_PORT:-8765}"
export FLEET_PIDFILE="${FLEET_PIDFILE:-${TMPDIR:-/tmp}/fleet-proxy.pid}"
export FLEET_LOGFILE="${FLEET_LOGFILE:-${TMPDIR:-/tmp}/fleet-proxy.log}"
export FLEET_API_KEY="${FLEET_API_KEY:-fleet-local}"

fleet-on() {
  if [ -f "$FLEET_PIDFILE" ] && kill -0 "$(cat "$FLEET_PIDFILE")" 2>/dev/null; then
    echo "fleet-proxy already running (pid $(cat "$FLEET_PIDFILE"), port $FLEET_PORT)"
  else
    if ! command -v fleet >/dev/null 2>&1; then
      echo "fleet-on: 'fleet' not on PATH — install with 'pip install -e .' from the repo" >&2
      return 1
    fi
    nohup fleet --serve --port "$FLEET_PORT" --api-key "$FLEET_API_KEY" \
      > "$FLEET_LOGFILE" 2>&1 &
    echo $! > "$FLEET_PIDFILE"
    printf "starting fleet-proxy (pid %s)" "$(cat "$FLEET_PIDFILE")"
    # Boot can take ~10s — first call loads sentence-transformers.
    for _ in $(seq 1 30); do
      if curl -sf "http://127.0.0.1:$FLEET_PORT/healthz" >/dev/null 2>&1; then
        echo " ready"
        break
      fi
      printf "."
      sleep 1
    done
    if ! curl -sf "http://127.0.0.1:$FLEET_PORT/healthz" >/dev/null 2>&1; then
      echo
      echo "fleet-on: proxy did not become ready in 30s — see $FLEET_LOGFILE" >&2
      return 1
    fi
  fi
  export ANTHROPIC_BASE_URL="http://localhost:$FLEET_PORT"
  export ANTHROPIC_API_KEY="$FLEET_API_KEY"
  echo "claude → fleet → Ollama (env set in this shell)"
}

fleet-off() {
  if [ -f "$FLEET_PIDFILE" ] && kill -0 "$(cat "$FLEET_PIDFILE")" 2>/dev/null; then
    kill "$(cat "$FLEET_PIDFILE")" && rm -f "$FLEET_PIDFILE"
    echo "fleet-proxy stopped"
  else
    rm -f "$FLEET_PIDFILE"  # stale pidfile cleanup
    echo "fleet-proxy was not running"
  fi
  unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY
  echo "claude → Anthropic (env unset)"
}

fleet-status() {
  if [ -f "$FLEET_PIDFILE" ] && kill -0 "$(cat "$FLEET_PIDFILE")" 2>/dev/null; then
    echo "proxy : RUNNING (pid $(cat "$FLEET_PIDFILE"), port $FLEET_PORT)"
  else
    echo "proxy : stopped"
  fi
  echo "shell : ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-(unset → Anthropic)}"
  echo "log   : $FLEET_LOGFILE"
}
