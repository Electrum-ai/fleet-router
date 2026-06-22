# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY fleet/ ./fleet/
COPY evals/ ./evals/

RUN pip install --prefix=/install -e ".[dev]"

# ── Stage 2: Production ─────────────────────────────────────────────────
FROM python:3.12-slim AS production

LABEL org.opencontainers.image.title="Fleet Router"
LABEL org.opencontainers.image.description="Adaptive parallel LLM router with verifier-driven synthesis for Ollama"
LABEL org.opencontainers.image.source="https://github.com/Electrum-ai/fleet-router"

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

RUN mkdir -p /root/.cache/huggingface && chmod 700 /root/.cache/huggingface
RUN mkdir -p /root/.fleet && chmod 700 /root/.fleet

EXPOSE 8765

# Default to serve mode. Override API_KEY via env.
CMD ["fleet", "--serve", "--host", "0.0.0.0", "--port", "8765", "--api-key", "change-me-in-env"]
