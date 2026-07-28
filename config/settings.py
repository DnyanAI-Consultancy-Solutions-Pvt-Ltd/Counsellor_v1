from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load application configuration from the .env file."""

    # Groq configuration
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_MODEL",
    )

    # Embedding and Chroma configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chroma_db_path: str = Field(
        default="storage/chroma",
        alias="CHROMA_DB_PATH",
    )
    collection_name: str = Field(
        default="mhtcet_cap_cutoffs",
        alias="COLLECTION_NAME",
    )

    # Retrieval configuration
    max_search_queries: int = Field(
        default=6,
        alias="MAX_SEARCH_QUERIES",
    )
    results_per_query: int = Field(
        default=12,
        alias="RESULTS_PER_QUERY",
    )
    retrieval_limit: int = Field(
        default=120,
        alias="RETRIEVAL_LIMIT",
    )

    # Context-size protection
    max_context_characters: int = Field(
        default=16000,
        alias="MAX_CONTEXT_CHARACTERS",
    )
    max_chars_per_document: int = Field(
        default=900,
        alias="MAX_CHARS_PER_DOCUMENT",
    )

    # Recommendation counts
    dream_count: int = Field(
        default=10,
        alias="DREAM_COUNT",
    )
    target_count: int = Field(
        default=10,
        alias="TARGET_COUNT",
    )
    safe_count: int = Field(
        default=10,
        alias="SAFE_COUNT",
    )
    recommendation_completion_attempts: int = Field(
        default=3,
        alias="RECOMMENDATION_COMPLETION_ATTEMPTS",
    )

    # Configuration-driven portfolio strategy. Margins are calculated as:
    # historical_cutoff - student_percentile.
    dream_margin_min: float = Field(default=0.5, alias="DREAM_MARGIN_MIN")
    dream_margin_max: float = Field(default=5.0, alias="DREAM_MARGIN_MAX")
    target_margin_min: float = Field(default=-1.5, alias="TARGET_MARGIN_MIN")
    target_margin_max: float = Field(default=0.5, alias="TARGET_MARGIN_MAX")
    safe_margin_min: float = Field(default=-5.0, alias="SAFE_MARGIN_MIN")
    safe_margin_max: float = Field(default=-1.5, alias="SAFE_MARGIN_MAX")

    dream_ratio: float = Field(default=0.30, alias="DREAM_RATIO")
    target_ratio: float = Field(default=0.40, alias="TARGET_RATIO")
    safe_ratio: float = Field(default=0.30, alias="SAFE_RATIO")

    # Processing configuration
    embedding_batch_size: int = Field(
        default=128,
        alias="EMBEDDING_BATCH_SIZE",
    )
    chroma_batch_size: int = Field(
        default=500,
        alias="CHROMA_BATCH_SIZE",
    )

    # LLM generation configuration
    llm_max_tokens: int = Field(
        default=3000,
        alias="LLM_MAX_TOKENS",
    )
    llm_temperature: float = Field(
        default=0.15,
        alias="LLM_TEMPERATURE",
    )

    # API configuration
    host: str = Field(
        default="127.0.0.1",
        alias="HOST",
    )
    port: int = Field(
        default=8000,
        alias="PORT",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    @property
    def recommendation_count(self) -> int:
        return self.dream_count + self.target_count + self.safe_count

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()