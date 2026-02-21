import pickle
from rank_bm25 import BM25Okapi
from src.config import settings


class BM25Index:
    def __init__(self):
        bm25_path = settings.index_dir / "bm25.pkl"
        print(f"[BM25] Loading index from {bm25_path}...")

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25: BM25Okapi = data["bm25"]
        self.metadata: list[dict] = data["metadata"]
        print(f"[BM25] Loaded {len(self.metadata)} chunks")

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = dict(self.metadata[idx])
            chunk["sparse_score"] = float(scores[idx])
            chunk["sparse_rank"] = rank
            results.append(chunk)

        return results


_bm25: BM25Index | None = None

def get_bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
    return _bm25
