import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings


class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Loading model on {self.device}...")
        self.model = SentenceTransformer(settings.embed_model, device=self.device)
        print(f"[Embedder] Ready on {self.device}")

    def encode(self, text: str | list[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings


# Singleton
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
