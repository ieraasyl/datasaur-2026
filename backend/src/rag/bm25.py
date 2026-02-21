"""BM25 sparse retriever for exact medical terminology matching."""
import logging
import pickle
import re

from rank_bm25 import BM25Okapi
from src.config import settings

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer; keeps ICD codes intact."""
    text = text.lower()
    tokens = re.split(r"\s+", text)
    # Remove pure punctuation tokens
    return [t.strip(".,;:!?()[]") for t in tokens if t.strip(".,;:!?()[]")]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: list[dict] = []  # parallel to BM25 corpus

    def build(self, chunks: list[dict]):
        """Build BM25 index from chunks."""
        tokenized = [_tokenize(c.get("chunk", c.get("text", ""))) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def save(self):
        """Save BM25 index to disk."""
        bm25_path = settings.index_dir / "bm25.pkl"
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.chunks}, f)
        logger.info(f"BM25 index saved → {bm25_path}")

    def load(self) -> bool:
        """Load BM25 index from disk."""
        bm25_path = settings.index_dir / "bm25.pkl"
        if not bm25_path.exists():
            return False
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data.get("metadata", data.get("chunks", []))
        logger.info(f"BM25 index loaded: {len(self.chunks)} documents")
        return True

    def search(self, query: str, top_k: int) -> list[dict]:
        """Search BM25 index and return top_k results."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call load() first.")
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            chunk = dict(self.chunks[idx])
            chunk["sparse_score"] = float(scores[idx])
            chunk["sparse_rank"] = rank
            results.append(chunk)
        return results


_bm25: BM25Index | None = None

def get_bm25() -> BM25Index:
    """Get singleton BM25Index instance, loading from disk if needed."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
        if not _bm25.load():
            logger.warning("BM25 index not found. Run index_corpus.py to build it.")
    return _bm25
