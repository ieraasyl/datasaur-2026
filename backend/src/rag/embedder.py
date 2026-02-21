"""Sentence-Transformers embedder with CUDA auto-detection.

Uses intfloat/multilingual-e5-small which requires:
  - corpus chunks prefixed with 'passage: '
  - queries prefixed with 'query: '
This is critical for correct retrieval quality with E5 models.
"""
import logging

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{settings.embed_model}' on device={device}")
        self._model = SentenceTransformer(settings.embed_model, device=device)
        logger.info("Embedding model loaded.")

    def encode(self, texts: list[str] | str, batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        """
        E5 models require prefixes:
          - passages (corpus chunks) : 'passage: <text>'
          - queries                  : 'query: <text>'
        Omitting the prefix degrades retrieval quality significantly.
        """
        self._load()
        if isinstance(texts, str):
            texts = [texts]

        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]

        vecs = self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,   # cosine via inner product on FAISS IndexFlatIP
            show_progress_bar=len(texts) > 100,
        )
        return np.array(vecs, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query string with the correct 'query: ' prefix."""
        return self.encode([query], is_query=True)[0]