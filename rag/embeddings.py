from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config.settings import get_settings


class EmbeddingService:
    """Shared sentence-transformer embedding service."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = SentenceTransformer(self.settings.embedding_model)

    @property
    def embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        clean = text.strip()
        if not clean:
            raise ValueError("Query text cannot be empty.")
        vector = self.model.encode(
            [clean],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
