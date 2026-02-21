from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent        # backend/
PROJECT_ROOT = BASE_DIR.parent                           # datasaur-2026/
_env_path = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path),
        env_file_encoding="utf-8",
    )

    gpt_oss_url: str = "https://hub.qazcode.ai"
    gpt_oss_api_key: str = ""
    gpt_oss_model: str = "oss-120b"

    index_dir: Path = BASE_DIR / "data" / "index"
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    static_dir: Path = PROJECT_ROOT / "static"

    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 7
    rrf_k: int = 60
    mock_llm: bool = False


settings = Settings()

TOP_K: int = settings.top_k
TOP_N_DIAG: int = 3

print(f"[Config] env_file={_env_path} exists={_env_path.exists()} api_key={'set' if settings.gpt_oss_api_key else 'EMPTY'}")
