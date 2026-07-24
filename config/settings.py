from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "MHT-CET Engineering CAP Round AI Counsellor"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    groq_api_key: str = ""
    admin_key: str = "change-this-admin-key"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    minimum_recommendations: int = 30
    desired_recommendations: int = 40
    retrieval_results_per_query: int = 14
    max_evidence_chunks: int = 120
    log_level: str = "INFO"
    storage_dir: Path = BASE_DIR / "storage"
    uploads_dir: Path = BASE_DIR / "storage" / "uploads"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    sessions_dir: Path = BASE_DIR / "storage" / "sessions"
    memory_file: Path = BASE_DIR / "storage" / "sessions.json"
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    def ensure_dirs(self) -> None:
        for directory in (self.storage_dir, self.uploads_dir, self.chroma_dir, self.sessions_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
