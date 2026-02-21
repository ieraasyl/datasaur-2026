"""Central config loaded from environment variables."""
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # backend/


class Settings(BaseSettings):
    # GPT-OSS (organizer LiteLLM proxy)
    gpt_oss_url: str = "https://hub.qazcode.ai"
    gpt_oss_api_key: str = ""
    gpt_oss_model: str = "oss-120b"
    mock_llm: bool = False  # Set to true for retrieval-only fallback

    # Paths
    index_dir: Path = BASE_DIR / "data" / "index"
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    static_dir: Path = BASE_DIR.parent / "static"  # Astro build output

    # Embedding model (multilingual-e5-small: ~120MB, excellent Russian support)
    embed_model: str = "intfloat/multilingual-e5-small"
    
    # Chunking parameters
    chunk_size: int = 600  # tokens (words)
    chunk_overlap: int = 100
    
    # Retrieval parameters
    top_k: int = 10  # chunks per retriever
    top_n_diag: int = 5  # diagnoses returned
    rrf_k: int = 60  # RRF constant
    
    # Reranker (cross-encoder for improved Accuracy@1)
    use_reranker: bool = True  # Enable cross-encoder reranker
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def gpt_oss_key(self) -> str:
        """Alias for backward compatibility."""
        return self.gpt_oss_api_key


settings = Settings()

# Convenience constants for backward compatibility
TOP_K = settings.top_k
TOP_N_DIAG = settings.top_n_diag
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MOCK_LLM = settings.mock_llm
GPT_OSS_URL = settings.gpt_oss_url
GPT_OSS_KEY = settings.gpt_oss_api_key
GPT_OSS_MODEL = settings.gpt_oss_model
EMBED_MODEL = settings.embed_model
INDEX_DIR = settings.index_dir
CORPUS_DIR = settings.corpus_dir
STATIC_DIR = settings.static_dir
FAISS_INDEX = INDEX_DIR / "faiss.index"
BM25_INDEX = INDEX_DIR / "bm25.pkl"
META_FILE = INDEX_DIR / "metadata.pkl"