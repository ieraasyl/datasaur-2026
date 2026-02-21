"""
index_corpus.py — One-time script to parse protocols, chunk, embed, and build
FAISS + BM25 indexes. Run from the backend/ directory:

    uv run python scripts/index_corpus.py [--corpus data/corpus] [--chunk-size 600]

For GPU-accelerated indexing, run on Colab/Kaggle and upload the index files
via GitHub Releases (see README).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Allow imports from backend/src
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Text chunking with protocol-aware splitting ──────────────────────────────

# Kazakh clinical protocol section headers — split on these for logical chunking
SECTION_RE = re.compile(
    r"(?:^|\n)(?=(?:I{1,3}V?|VI{0,3}|[1-9]\d*)\.\s+[А-ЯA-Z])",
    re.MULTILINE,
)

# Diagnostic criteria keywords (high-value sections for symptom matching)
DIAGNOSTIC_KEYWORDS = [
    "диагностические критерии",
    "жалобы",
    "лабораторные исследования",
    "клинические признаки",
    "симптомы",
    "диагностика",
    "критерии диагноза",
]


def chunk_by_sections(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split on protocol section headers first, then by word count.
    Prioritizes diagnostic criteria sections for better symptom matching.
    """
    # First, try to split on section headers
    sections = SECTION_RE.split(text)
    sections = [s.strip() for s in sections if s.strip()]
    
    if not sections:
        # Fallback: treat entire text as one section
        sections = [text]

    chunks: list[str] = []
    for section in sections:
        # Check if this section contains diagnostic keywords (higher priority)
        section_lower = section.lower()
        is_diagnostic_section = any(kw in section_lower for kw in DIAGNOSTIC_KEYWORDS)
        
        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            # Sliding window within long sections
            # Use smaller overlap for diagnostic sections to preserve more context
            section_overlap = overlap // 2 if is_diagnostic_section else overlap
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start += chunk_size - section_overlap
    
    return chunks


def extract_icd_from_text(text: str) -> list[str]:
    """Extract ICD-10 codes from text using regex."""
    return list(set(re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", text)))


def load_protocols(corpus_path: Path) -> list[dict]:
    """Load protocols from JSON/JSONL files."""
    protocols: list[dict] = []
    # Support .jsonl files (one JSON object per line)
    for fpath in sorted(corpus_path.glob("**/*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    protocols.append(json.loads(line))
    # Support single .json list files
    for fpath in sorted(corpus_path.glob("**/*.json")):
        # Skip if it's a test file or response file
        if "test" in fpath.name.lower() or "response" in fpath.name.lower():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                protocols.extend(data)
            elif isinstance(data, dict):
                protocols.append(data)
    logger.info(f"Loaded {len(protocols)} protocols from {corpus_path}")
    return protocols


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus", help="Corpus directory")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size (words)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap (words)")
    args = parser.parse_args()

    from src.config import settings
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import pickle
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        corpus_path = settings.corpus_dir
    if not corpus_path.exists():
        logger.error(f"Corpus path not found: {corpus_path}")
        sys.exit(1)

    # Load protocols
    protocols = load_protocols(corpus_path)
    if not protocols:
        logger.error("No protocols found! Check corpus path and file format.")
        sys.exit(1)

    # Chunk all protocols
    all_chunks: list[dict] = []
    for proto in protocols:
        pid = proto.get("protocol_id", "")
        src = proto.get("source_file", "")
        title = proto.get("title", "")
        icds = proto.get("icd_codes", [])
        text = proto.get("text", "")

        # Merge ICD codes from text as well
        found_icds = extract_icd_from_text(text)
        all_icds = list(set(icds + found_icds))

        for idx, chunk in enumerate(chunk_by_sections(text, args.chunk_size, args.overlap)):
            all_chunks.append({
                "protocol_id": pid,
                "source_file": src,
                "title": title,
                "icd_codes": all_icds,
                "chunk": chunk,
                "chunk_idx": idx,
                "chunk_index": idx,  # Support both field names
                "text": chunk,  # Support both field names
            })

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Embed all chunks using the Embedder class (handles E5 prefixes correctly)
    logger.info("Embedding chunks (may take several minutes on CPU)...")
    from src.rag.embedder import Embedder
    embedder = Embedder()
    texts = [c["chunk"] for c in all_chunks]
    # Embedder.encode() automatically adds 'passage: ' prefix for corpus chunks
    embeddings = embedder.encode(texts, batch_size=64, is_query=False)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build & save FAISS index
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    logger.info(f"✅ FAISS index saved: {index.ntotal} vectors (dim={dim})")

    # Build & save BM25 index
    logger.info("Building BM25 index...")
    from src.rag.bm25 import _tokenize
    tokenized = [_tokenize(c.get("chunk", c.get("text", ""))) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": all_chunks}, f)
    logger.info(f"✅ BM25 index saved: {len(all_chunks)} documents")

    # Save metadata
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info("✅ Metadata saved")

    logger.info(f"\n✅ Indexing complete! Indexes saved to {index_dir}")
    for f in index_dir.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info(f"   {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
