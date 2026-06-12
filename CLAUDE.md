# Fleet Router — Claude Code integration notes

This file is auto-loaded by Claude Code when it opens this project.

## What's wired up

When Claude Code starts a session in this directory, the SessionStart hook
at `.claude/settings.json` runs `scripts/fleet-ensure-proxy.py`. That
script:

- Checks `~/.fleet/run/fleet-proxy.pid` and `http://127.0.0.1:8765/healthz`
- Starts `venv/bin/fleet --serve --port 8765 --api-key fleet-local` if not up
- Coordinates concurrent SessionStart fires via flock on
  `~/.fleet/run/fleet-ensure-proxy.lock`
- Polls `/healthz` for up to 60s (cold start loads sentence-transformers)

Runtime state lives in a private `~/.fleet/run/` directory (created mode
0700) — not world-writable `$TMPDIR`/`/tmp`, where a predictable path
allowed symlink and pid-confusion attacks. Logs land in
`~/.fleet/run/fleet-proxy.log`.

The hook only ensures the proxy is running. It does **not** redirect this
Claude Code session through the proxy — see "Tool-loop limitation" below
for why.

## Using `/fleet` from any chat

The slash command is installed at `~/.claude/skills/fleet/SKILL.md`. From
any chat (any directory):

```
/fleet "write a Python function that parses TOML"
/fleet --parallel "compare these approaches"
/fleet --model glm-5.1 "translate this paragraph"
```

`/fleet` calls the `fleet` CLI directly, not the HTTP proxy.

## Routing aider through Fleet (opt-in, scoped)

aider speaks the OpenAI Chat Completions API, which the proxy now
implements at `POST /v1/chat/completions`. Point aider at it via env
vars and pick any model name fleet routes (the registry is exposed at
`GET /v1/models`):

```bash
# Make sure the proxy is up (the SessionStart hook does this in this dir,
# or run scripts/fleet-ensure-proxy.py manually elsewhere).

export OPENAI_API_BASE=http://localhost:8765/v1
export OPENAI_API_KEY=fleet-local

aider --model openai/deepseek-v4-flash       # fast single-model
aider --model openai/deepseek-v4-pro         # max-quality reasoning
```

Notes:

- aider prefixes model names with `openai/` to tell LiteLLM which
  provider format to use; fleet's proxy ignores the prefix and routes
  by the bare model name.
- Streaming and non-streaming both work. Heartbeats are SSE comment
  lines (`: keep-alive`) which aider's parser silently skips, so
  60–180s max-quality runs don't trigger the connection-reset bug.
- The `--model openai/X` you pass is echoed back in `model` fields and
  passed through to the underlying ollama:cloud route via fleet's
  registry. To force a single model regardless of what aider sends,
  edit `~/.fleet/config.yaml` to mark only one entry with `priority: 1`
  and matching tags.
- **Same tool-loop limitation as below**: aider's `/edit` and shell
  modes use function-calling under the hood. Function calls are
  flattened to text, so the model will *describe* edits rather than
  apply them. For pure code-generation chat (`/ask`, `/code` with
  manual paste), aider works great. For agentic editing, stay on
  Anthropic/OpenAI direct.

## Routing Claude Code itself through Fleet (opt-in, scoped)

The proxy implements enough of the Anthropic Messages API for plain chat
to work, but **tool blocks (`tools` / `tool_use` / `tool_result`) are
flattened to text** — see the docstring at the top of `fleet/proxy.py`.
That means Read, Edit, Bash, and every other Claude Code tool will not
function when routed through Fleet. The model will describe what it would
do instead of doing it.

For that reason this project's `.claude/settings.json` does **not** set
`ANTHROPIC_BASE_URL` or `ANTHROPIC_API_KEY`. Claude Code in this directory
behaves normally (Anthropic backend, full tools).

To opt in for a single shell — chat-only, no tools — source the toggle:

```bash
source scripts/fleet-toggle.sh
fleet-on            # sets ANTHROPIC_* in this shell, boots the proxy
claude              # Claude Code launched from this shell uses Fleet → Ollama
fleet-off           # restore Anthropic and stop the proxy
fleet-status        # check current state
```

The env vars are scoped to that one shell. GUI-launched Claude Code
windows still use Anthropic.

## Troubleshooting

- **Hook never ran**: check `~/.claude/settings.json` doesn't override the
  project hook. Settings merge — project additions, user overrides.
- **Proxy didn't come up**: read `~/.fleet/run/fleet-proxy.log` and run
  `python3 scripts/fleet-ensure-proxy.py` manually to see stderr.
- **`(all models failed)` in chat replies**: Ollama isn't reachable. The
  proxy now appends a hint with `ollama serve` and `curl
  http://localhost:11434/api/tags` checks (`fleet/proxy.py`
  `_maybe_enrich_with_ollama_hint`).
- **Port 8765 already taken** by an unrelated process: the ensure-proxy
  script will refuse to auto-kill it. Stop that process or set
  `FLEET_PORT=<other>` in your environment.

## Rollback

- Disable the slash command: `rm ~/.claude/skills/fleet/SKILL.md`
- Disable the auto-start: `rm /Users/bistrocloud/fleet-router/.claude/settings.json`
- Stop the proxy: `kill $(cat ~/.fleet/run/fleet-proxy.pid)` or
  `fleet-off` if the toggle is sourced.

## File map for this integration

- `fleet/SKILL.md` — slash-command spec (mirrored to `~/.claude/skills/fleet/`)
- `fleet/proxy.py` — HTTP proxy speaking both Anthropic
  (`/v1/messages`) and OpenAI (`/v1/chat/completions`) dialects, plus
  `/v1/models` and `/healthz`
- `scripts/fleet-toggle.sh` — manual shell-scoped opt-in
- `scripts/fleet-ensure-proxy.py` — idempotent background-boot for the
  SessionStart hook
- `.claude/settings.json` — wires the SessionStart hook
- `tests/test_proxy.py` — proxy contract tests

## Project memory

User memory (loaded automatically) records:

- This project is Ollama-only — no OpenAI/Anthropic/proprietary providers.
- Quality > speed/cost; verifiers + self-consistency + abstention are
  defaults.
- See `~/.claude/projects/-Users-bistrocloud-fleet-router/memory/` for the
  full set.
