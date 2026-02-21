# ─── Stage 1: Build Astro frontend ───────────────────────────────────────────
FROM oven/bun:1 AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/bun.lock* ./
RUN bun install --frozen-lockfile || bun install

COPY frontend/ .
RUN bun run build
# astro.config.mjs: outDir: "../static" → output lands in /app/static


# ─── Stage 2: Python backend + CUDA-aware runtime ─────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Python dependencies — use pyproject.toml as the source of truth
COPY backend/pyproject.toml ./pyproject.toml
RUN uv venv .venv --python python3.12 && \
    uv pip install --python .venv/bin/python -r pyproject.toml

# Backend source (selective copy to avoid bloating the image with local data/)
COPY backend/src ./src
COPY backend/scripts ./scripts

# ── Download pre-built index from GitHub Releases ────────────────────────────
ARG INDEX_RELEASE_URL=""

RUN mkdir -p data/index && \
    if [ -n "$INDEX_RELEASE_URL" ]; then \
        echo "Downloading pre-built index from: $INDEX_RELEASE_URL" && \
        curl -L "$INDEX_RELEASE_URL" -o /tmp/index.zip && \
        unzip /tmp/index.zip -d data/index/ && \
        rm /tmp/index.zip && \
        echo "Index downloaded successfully."; \
    else \
        echo "NOTE: INDEX_RELEASE_URL not set — will try local corpus fallback."; \
    fi

# Fallback: copy corpus and build index at image time if no pre-built index
COPY backend/data/corpus ./data/corpus
RUN if [ ! -f "data/index/faiss.index" ] && [ -d "data/corpus" ] && [ "$(ls -A data/corpus 2>/dev/null)" ]; then \
        echo "Building indexes from corpus (slow on CPU — prefer Colab + INDEX_RELEASE_URL)..." && \
        .venv/bin/python scripts/index_corpus.py; \
    elif [ ! -f "data/index/faiss.index" ]; then \
        echo "WARNING: No index and no corpus. Service will start in degraded mode."; \
    fi

# Frontend static build
COPY --from=frontend-builder /app/static ./static/

EXPOSE 8080

# Override config.py defaults for the container layout (/app as root)
ENV INDEX_DIR=/app/data/index
ENV CORPUS_DIR=/app/data/corpus
ENV STATIC_DIR=/app/static

CMD [".venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
