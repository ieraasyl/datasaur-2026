# Datasaur 2026 — Medical Diagnosis Assistant

AI-powered clinical decision support system using Kazakhstan clinical protocols (МКБ-10).

## Architecture

```
Symptoms (text) 
  → Embed (multilingual-e5-small)
  → Hybrid Retrieval (FAISS dense + BM25 sparse + RRF fusion)
  → Cross-Encoder Re-ranking (mmarco-mMiniLMv2, optional)
  → GPT-OSS (oss-120b) with ICD-10 constrained prompt
  → Ranked diagnoses + ICD-10 codes
```

**Key improvements:**
- **Better embeddings**: `intfloat/multilingual-e5-small` (~120MB, excellent Russian support)
- **Cross-encoder reranker**: Improves Accuracy@1 by re-scoring top chunks
- **Protocol-aware chunking**: Prioritizes diagnostic criteria sections
- **ICD-10 constrained prompts**: Reduces hallucinated codes

Single Docker container on port 8080. FastAPI backend serves Astro static frontend.

## Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- OR Docker without GPU (falls back to CPU automatically)
- `bun` and `uv` for local development

## Setup

### 1. Clone and configure

```bash
git clone <your-repo>
cd datasaur-2026
cp .env.example .env
# Fill in GPT_OSS_API_KEY in .env
```

### 2. Extract corpus

```bash
# Put corpus.zip contents into backend/data/corpus/
unzip corpus.zip -d backend/data/corpus/
```

### 3. Build indexes (one-time)

**Option A: Build on GPU (Recommended — much faster)**

For best performance, build indexes on Colab/Kaggle with GPU:

1. Upload `backend/scripts/index_corpus.py` and corpus to Colab/Kaggle
2. Install dependencies: `pip install sentence-transformers faiss-cpu rank-bm25 torch numpy`
3. Run: `python index_corpus.py --corpus ./corpus`
4. Download the generated `data/index/` folder
5. Place it in `backend/data/index/` in your local repo

**Option B: Build locally (CPU — slower but works)**

```bash
cd backend
uv venv && source .venv/bin/activate
uv sync
python scripts/index_corpus.py
# Creates: backend/data/index/faiss.index, bm25.pkl, metadata.pkl
```

**Option C: Use pre-built indexes from GitHub Releases**

If indexes are too large for git, download from GitHub Releases and extract to `backend/data/index/`.

### 4. Validate response format early

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8080 &
uv run python ../evaluate.py -e http://127.0.0.1:8080/diagnose -d ../data/test_set -n FCB
```

## Local Development

```bash
docker compose up
# Backend: http://localhost:8080
# Frontend dev: http://localhost:4321
```

## Build & Submit

```bash
docker build -t submission .

# With GPU
docker run --gpus all -p 8080:8080 --env-file .env submission

# Without GPU
docker run -p 8080:8080 --env-file .env submission
```

## Evaluate

```bash
uv run python evaluate.py -e http://127.0.0.1:8080/diagnose -d ./data/test_set -n FCB
```

## Project Structure

```
datasaur-2026/
├── frontend/          # Astro + React UI (builds to /static)
├── backend/
│   ├── src/           # FastAPI app + RAG pipeline
│   ├── scripts/       # index_corpus.py (run once)
│   └── data/
│       ├── corpus/    # Extracted protocol JSONs
│       └── index/     # Pre-built FAISS + BM25 indexes
├── data/test_set/     # From hack-nu repo
├── evaluate.py        # From hack-nu repo
├── Dockerfile
└── docker-compose.yml
```
