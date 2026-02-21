"""FAISS dense vector store."""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        """Build FAISS index from embeddings and metadata."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine (vecs already normalized)
        self.index.add(embeddings)
        self.metadata = metadata
        logger.info(f"FAISS index built: {self.index.ntotal} vectors (dim={dim})")

    def save(self):
        """Save FAISS index and metadata to disk."""
        index_path = settings.index_dir / "faiss.index"
        meta_path = settings.index_dir / "metadata.pkl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"FAISS index saved → {index_path}")

    def load(self) -> bool:
        """Load FAISS index and metadata from disk."""
        index_path = settings.index_dir / "faiss.index"
        meta_path = settings.index_dir / "metadata.pkl"
        if not index_path.exists() or not meta_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Returns list of chunk dicts with added 'score' and 'dense_rank' fields."""
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load() first.")
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            chunk = dict(self.metadata[idx])
            chunk["dense_score"] = float(score)
            chunk["dense_rank"] = rank
            results.append(chunk)
        return results


_store: VectorStore | None = None

def get_vectorstore() -> VectorStore:
    """Get singleton VectorStore instance, loading from disk if needed."""
    global _store
    if _store is None:
        _store = VectorStore()
        if not _store.load():
            logger.warning("FAISS index not found. Run index_corpus.py to build it.")
    return _store
