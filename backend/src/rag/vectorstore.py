import faiss
import numpy as np
import pickle
from pathlib import Path
from src.config import settings


class VectorStore:
    def __init__(self):
        index_path = settings.index_dir / "faiss.index"
        meta_path = settings.index_dir / "metadata.pkl"

        print(f"[VectorStore] Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)  # list of dicts per chunk

        print(f"[VectorStore] Loaded {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["dense_score"] = float(score)
            chunk["dense_rank"] = len(results) + 1
            results.append(chunk)

        return results


_store: VectorStore | None = None

def get_vectorstore() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
