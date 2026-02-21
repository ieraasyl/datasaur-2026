import numpy as np
from src.rag.vectorstore import get_vectorstore
from src.rag.bm25 import get_bm25
from src.config import settings


def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for rank, chunk in enumerate(dense, start=1):
        cid = f"{chunk['protocol_id']}_{chunk['chunk_index']}"
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(sparse, start=1):
        cid = f"{chunk['protocol_id']}_{chunk['chunk_index']}"
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunks_by_id[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        chunk = dict(chunks_by_id[cid])
        chunk["rrf_score"] = score
        chunk["rrf_rank"] = rank
        results.append(chunk)

    return results


def hybrid_search(query: str, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.top_k

    dense_results = get_vectorstore().search(query_embedding, top_k=top_k)
    sparse_results = get_bm25().search(query, top_k=top_k)

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=settings.rrf_k)
    return fused[:top_k]
