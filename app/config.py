"""RemedyRadar configuration — loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings, populated from .env or environment variables."""

    # ── LLM (Cloud API) ──
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama3-8b-8192"
    embed_model: str = "nomic-embed-text-v1_5"

    # ── Database ──
    database_url: str = "postgresql+asyncpg://remedy:remedy@localhost:5432/remedyradar"

    # ── ChromaDB ──
    chroma_persist_dir: str = "./chroma_data"

    # ── Search Defaults ──
    default_search_radius_km: float = 10.0
    max_search_radius_km: float = 50.0
    top_n_doctors: int = 5

    # ── Logging ──
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
