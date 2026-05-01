"""
Application settings — loads from .env on every instantiation.

Key design decisions:
  - load_dotenv() with override=True is called at module-import time so that
    .env values always win over any stale shell-level environment variables.
  - get_settings() has NO lru_cache: every call returns a fresh Settings
    object, guaranteeing .env changes are picked up without restarting.
  - The .env path is resolved relative to THIS file so the server finds it
    regardless of which working directory it is launched from.
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to .env — works no matter where uvicorn is started from
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Force-load .env into os.environ right now, with override=True so that
# the real values always win over any empty/stale shell env vars.
load_dotenv(dotenv_path=_ENV_FILE, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,   # skip empty-string env vars, use .env value
        extra="ignore",
    )

    # Google OAuth
    google_client_id:     str = ""
    google_client_secret: str = ""
    google_redirect_uri:  str = "http://localhost:8000/auth/google/callback"
    frontend_url:         str = "http://localhost:8501"

    # LLM
    groq_api_key: str = ""
    groq_model:   str = "llama-3.3-70b-versatile"

    # Security
    secret_key:                  str = "change-me-in-production"
    access_token_expire_minutes: int = 1440
    algorithm:                   str = "HS256"

    # Storage
    chroma_db_path:  str = "./vectorstore"
    documents_path:  str = "./data/documents"
    database_url:    str = "sqlite:///./app.db"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    hf_token: str = ""
    hf_hub_disable_xet: bool = True

    # RAG
    top_k_results:        int   = 5
    similarity_threshold: float = 0.3
    chunk_size:           int   = 500
    chunk_overlap:        int   = 50
    pdf_ingestion_mode:   str   = "auto"
    enable_ocr_fallback:  bool  = True
    enable_summary_index: bool  = True
    enable_section_summaries: bool = False
    enable_image_summaries: bool = True
    summary_top_k:        int   = 3
    summary_max_chars:    int   = 12000
    vision_model:         str   = "meta-llama/llama-4-scout-17b-16e-instruct"
    vision_max_images_per_document: int = 5
    grounding_min_ratio:  float = 0.12

    # Voice
    whisper_model: str = "base"

    # API
    api_host:      str = "0.0.0.0"
    api_port:      int = 8000
    frontend_port: int = 8501


def get_settings() -> Settings:
    """
    Return a fresh Settings instance on every call.
    No caching — guarantees that .env changes are always reflected.
    """
    return Settings()
