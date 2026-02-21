from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # backend/

class Settings(BaseSettings):
    gpt_oss_url: str = "https://hub.qazcode.ai"
    gpt_oss_api_key: str = ""
    gpt_oss_model: str = "oss-120b"

    index_dir: Path = BASE_DIR / "data" / "index"
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    static_dir: Path = BASE_DIR.parent / "static"  # Astro build output

    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 10
    rrf_k: int = 60

    class Config:
        env_file = ".env"

settings = Settings()
