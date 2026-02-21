"""Hybrid retriever: FAISS (dense) + BM25 (sparse) fused via Reciprocal Rank Fusion."""
import logging
from collections import defaultdict

import numpy as np

from src.config import settings
from src.rag.bm25 import BM25Index
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant — standard value from the 2009 paper


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Calculate Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists using RRF.
    Deduplicates by (protocol_id, chunk_index/chunk_idx).
    Returns top_k fused results sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results):
        chunk_idx = chunk.get("chunk_index", chunk.get("chunk_idx", 0))
        key = f"{chunk['protocol_id']}:{chunk_idx}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, k)
        items[key] = chunk

    for rank, chunk in enumerate(sparse_results):
        chunk_idx = chunk.get("chunk_index", chunk.get("chunk_idx", 0))
        key = f"{chunk['protocol_id']}:{chunk_idx}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, k)
        if key not in items:
            items[key] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**items[k], "rrf_score": s} for k, s in ranked]


def aggregate_by_protocol(chunks: list[dict], top_protocols: int = 5) -> list[dict]:
    """
    Aggregate chunk-level scores by protocol_id.
    Returns chunks re-ordered so that chunks from the highest-scoring protocols
    come first, preserving intra-protocol chunk ordering by score.
    """
    protocol_data: dict[str, dict] = defaultdict(
        lambda: {"total_score": 0.0, "max_score": 0.0, "chunk_count": 0, "chunks": []}
    )

    for c in chunks:
        pid = c["protocol_id"]
        score = c.get("reranker_score", c.get("rrf_score", 0.0))
        protocol_data[pid]["total_score"] += score
        protocol_data[pid]["max_score"] = max(protocol_data[pid]["max_score"], score)
        protocol_data[pid]["chunk_count"] += 1
        protocol_data[pid]["chunks"].append(c)

    ranked_protocols = sorted(
        protocol_data.items(),
        key=lambda x: (x[1]["max_score"], x[1]["total_score"]),
        reverse=True,
    )[:top_protocols]

    result: list[dict] = []
    for pid, data in ranked_protocols:
        sorted_chunks = sorted(
            data["chunks"],
            key=lambda c: c.get("reranker_score", c.get("rrf_score", 0.0)),
            reverse=True,
        )
        for c in sorted_chunks:
            c["protocol_rank_score"] = data["max_score"]
        result.extend(sorted_chunks)

    logger.debug(
        f"Protocol aggregation: {len(chunks)} chunks → "
        f"{len(ranked_protocols)} protocols → {len(result)} chunks"
    )
    return result


class HybridRetriever:
    """Hybrid retriever combining dense (FAISS) and sparse (BM25) search."""

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vs = vector_store
        self.bm25 = bm25_index

    def search(self, query: str, query_embedding: np.ndarray, k: int) -> list[dict]:
        """Perform hybrid search and return fused results."""
        dense_results = self.vs.search(query_embedding, top_k=k)
        sparse_results = self.bm25.search(query, top_k=k)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=k, k=settings.rrf_k)
        logger.debug(f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused)} fused")
        return fused


def hybrid_search(query: str, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
    """Convenience function for hybrid search using singleton instances."""
    from src.rag.vectorstore import get_vectorstore
    from src.rag.bm25 import get_bm25
    
    top_k = top_k or settings.top_k
    retriever = HybridRetriever(get_vectorstore(), get_bm25())
    return retriever.search(query, query_embedding, k=top_k)
