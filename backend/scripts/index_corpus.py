"""
Run once to build FAISS + BM25 indexes from the corpus.

Usage:
    cd datasaur-2026/backend
    uv run python scripts/index_corpus.py
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Paths
BACKEND_DIR = Path(__file__).parent.parent
CORPUS_DIR = BACKEND_DIR / "data" / "corpus"
INDEX_DIR = BACKEND_DIR / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 600   # tokens approx (chars / 4)
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-based chunking with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def load_corpus() -> list[dict]:
    protocols = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            protocols.extend(data)
        else:
            protocols.append(data)

    for path in sorted(CORPUS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    protocols.append(json.loads(line))

    print(f"Loaded {len(protocols)} protocols")
    return protocols


def build_chunks(protocols: list[dict]) -> list[dict]:
    chunks = []
    for protocol in protocols:
        text = protocol.get("text", "")
        if not text.strip():
            continue
        for i, chunk_text_str in enumerate(chunk_text(text)):
            chunks.append({
                "protocol_id": protocol.get("protocol_id", ""),
                "source_file": protocol.get("source_file", ""),
                "title": protocol.get("title", ""),
                "icd_codes": protocol.get("icd_codes", []),
                "chunk_index": i,
                "text": chunk_text_str,
            })
    print(f"Built {len(chunks)} chunks from {len(protocols)} protocols")
    return chunks


def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> None:
    print("Building FAISS index...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"Saved FAISS index: {index.ntotal} vectors, dim={dim}")


def build_bm25_index(chunks: list[dict]) -> None:
    print("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": chunks}, f)
    print(f"Saved BM25 index: {len(chunks)} chunks")


def save_metadata(chunks: list[dict]) -> None:
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Saved metadata.pkl")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    protocols = load_corpus()
    if not protocols:
        print("ERROR: No protocols found in", CORPUS_DIR)
        sys.exit(1)

    chunks = build_chunks(protocols)
    build_faiss_index(chunks, model)
    build_bm25_index(chunks)
    save_metadata(chunks)

    print("\nDone! Index files written to:", INDEX_DIR)
    for f in INDEX_DIR.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
