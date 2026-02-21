# ── Stage 1: Build Astro frontend ──────────────────────────────────────────
FROM oven/bun:1 AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/bun.lock* ./
RUN bun install --frozen-lockfile
COPY frontend/ .
RUN bun run build
# Output goes to /app/static (see astro.config.mjs outDir: "../static")

# ── Stage 2: Python backend + CUDA ─────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps first (cache layer)
COPY backend/pyproject.toml ./
RUN uv venv .venv --python python3.12 && \
    uv pip install --system -r pyproject.toml 2>/dev/null || \
    uv sync --no-dev 2>/dev/null || \
    pip3 install fastapi uvicorn openai sentence-transformers faiss-cpu rank-bm25 pydantic pydantic-settings python-dotenv torch numpy tqdm

# Copy backend source
COPY backend/ .

# Copy pre-built frontend static files
COPY --from=frontend-builder /app/static ./static/

# Indexes must be pre-built and committed (or built in CI)
# They live at backend/data/index/ → /app/data/index/

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
