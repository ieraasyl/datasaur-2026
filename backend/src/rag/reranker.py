"""Cross-encoder re-ranker for improving retrieval accuracy."""
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# Enable/disable reranker (from settings)
USE_RERANKER = settings.use_reranker


class CrossEncoderReranker:
    """Cross-encoder re-ranker for improving Accuracy@1 by re-scoring retrieved chunks."""

    def __init__(self):
        self._model = None
        self._device = None

    def _load(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading cross-encoder model '{settings.reranker_model}' on device={self._device}...")
            self._model = CrossEncoder(settings.reranker_model, device=self._device)
            logger.info("Cross-encoder model loaded.")
        except ImportError:
            logger.warning("sentence-transformers not available for reranker. Install with: pip install sentence-transformers")
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}. Reranking disabled.")
            self._model = None

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """
        Re-rank chunks using cross-encoder.
        
        Args:
            query: User query/symptoms
            chunks: List of retrieved chunks (from hybrid search)
            top_k: Number of top chunks to return after re-ranking
            
        Returns:
            Re-ranked list of chunks, sorted by cross-encoder score (descending)
        """
        if not chunks:
            return chunks
        
        self._load()
        
        if self._model is None:
            logger.debug("Reranker model not available, returning original ranking.")
            return chunks[:top_k]
        
        try:
            # Prepare query-chunk pairs for cross-encoder
            # Handle both 'chunk' and 'text' field names
            chunk_texts = [
                c.get("chunk", c.get("text", ""))
                for c in chunks
            ]
            
            # Cross-encoder expects list of (query, passage) pairs
            pairs = [(query, text) for text in chunk_texts]
            
            # Get relevance scores (higher = more relevant)
            scores = self._model.predict(pairs)
            
            # Combine scores with original chunks and sort
            scored_chunks = [
                {**chunk, "reranker_score": float(score)}
                for chunk, score in zip(chunks, scores)
            ]
            
            # Sort by reranker score (descending)
            scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
            
            logger.debug(f"Re-ranked {len(chunks)} chunks, returning top {top_k}")
            return scored_chunks[:top_k]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original ranking.")
            return chunks[:top_k]
