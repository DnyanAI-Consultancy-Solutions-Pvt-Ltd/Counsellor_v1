from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import chromadb

from config.settings import get_settings
from rag.chunker import DocumentChunker, TextChunk
from rag.embeddings import get_embedding_service
from rag.loader import DocumentLoader


logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedding_service = get_embedding_service()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_db_path)
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def document_count(self) -> int:
        return self.collection.count()

    def health(self) -> dict[str, Any]:
        return {
            "collection": self.collection.name,
            "chunks": self.document_count,
            "storage_path": self.settings.chroma_db_path,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.embedding_service.embedding_dimension,
        }

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.settings.collection_name)
        except Exception:
            pass
        self.collection = self._get_collection()

    def index_document(
        self,
        file_path: str | Path,
        document_type: str = "general",
    ) -> dict[str, Any]:
        path = Path(file_path)
        started = time.perf_counter()

        logger.info("STEP 1/4: Loading document %s", path.name)
        pages = self.loader.load(path)
        logger.info("Loaded %s pages in %.2f seconds", len(pages), time.perf_counter() - started)

        chunk_started = time.perf_counter()
        logger.info("STEP 2/4: Parsing and grouping cutoff records")
        chunks = self.chunker.chunk_pages(
            pages,
            base_metadata={"source_file": path.name, "document_type": document_type},
        )
        logger.info(
            "Created %s chunks in %.2f seconds",
            len(chunks),
            time.perf_counter() - chunk_started,
        )

        index_started = time.perf_counter()
        logger.info("STEP 3/4: Generating embeddings and writing to Chroma")
        result = self.index_chunks(chunks)
        logger.info(
            "Embedding and indexing completed in %.2f seconds",
            time.perf_counter() - index_started,
        )

        total_time = time.perf_counter() - started
        logger.info("STEP 4/4: Completed %s in %.2f seconds", path.name, total_time)

        return {
            "file_name": path.name,
            "document_type": document_type,
            "pages_loaded": len(pages),
            "chunks_created": len(chunks),
            "structured_cutoff_records": sum(
                1
                for chunk in chunks
                if chunk.metadata.get("record_type") == "cap_branch_cutoff"
            ),
            "processing_seconds": round(total_time, 2),
            **result,
        }

    def index_chunks(self, chunks: list[TextChunk]) -> dict[str, int]:
        if not chunks:
            return {"indexed": 0, "duplicates": 0}

        ids = [self._chunk_id(chunk) for chunk in chunks]
        existing: set[str] = set()

        for start in range(0, len(ids), 500):
            result = self.collection.get(ids=ids[start:start + 500], include=[])
            existing.update(result.get("ids") or [])

        documents = []
        metadatas = []
        new_ids = []
        duplicates = 0

        for chunk, chunk_id in zip(chunks, ids):
            if chunk_id in existing:
                duplicates += 1
                continue
            if not chunk.text.strip():
                continue

            documents.append(chunk.text.strip())
            metadatas.append(self._clean_metadata(chunk.metadata))
            new_ids.append(chunk_id)

        batch_size = self.settings.chroma_batch_size
        total_batches = (
            (len(documents) + batch_size - 1) // batch_size
            if documents else 0
        )

        for batch_number, start in enumerate(
            range(0, len(documents), batch_size),
            start=1,
        ):
            batch_started = time.perf_counter()
            docs = documents[start:start + batch_size]

            logger.info(
                "Embedding batch %s/%s (%s records)",
                batch_number,
                total_batches,
                len(docs),
            )
            embeddings = self.embedding_service.embed_documents(docs)

            logger.info("Writing batch %s/%s to Chroma", batch_number, total_batches)
            self.collection.add(
                ids=new_ids[start:start + batch_size],
                documents=docs,
                metadatas=metadatas[start:start + batch_size],
                embeddings=embeddings,
            )
            logger.info(
                "Completed batch %s/%s in %.2f seconds",
                batch_number,
                total_batches,
                time.perf_counter() - batch_started,
            )

        return {"indexed": len(new_ids), "duplicates": duplicates}

    def search(
        self,
        query: str,
        top_k: int = 20,
        document_type: str | None = None,
        source_file: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip() or self.document_count == 0:
            return []

        filters = []
        if document_type:
            filters.append({"document_type": {"$eq": document_type}})
        if source_file:
            filters.append({"source_file": {"$eq": source_file}})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        kwargs: dict[str, Any] = {
            "query_embeddings": [self.embedding_service.embed_query(query)],
            "n_results": min(top_k, self.document_count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        return [
            {
                "text": doc or "",
                "metadata": meta or {},
                "distance": float(distance),
                "similarity": round(1.0 - float(distance), 6),
            }
            for doc, meta, distance in zip(docs, metas, distances)
        ]


    def list_cutoff_records(self, document_type: str = "cutoff") -> list[dict[str, Any]]:
        """Return all indexed structured cutoff records for deterministic filtering.

        Semantic search is still used for ranking/relevance, but this exhaustive pool
        prevents valid colleges from being lost merely because they were outside a
        small vector-search top-k result.
        """
        if self.document_count == 0:
            return []

        where = {"document_type": {"$eq": document_type}} if document_type else None
        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
            "limit": self.document_count,
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.get(**kwargs)
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        ids = raw.get("ids") or []

        output: list[dict[str, Any]] = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            metadata = meta or {}
            if metadata.get("record_type") != "cap_branch_cutoff":
                continue
            output.append({
                "id": chunk_id,
                "text": doc or "",
                "metadata": metadata,
                "distance": 1.0,
                "similarity": 0.0,
            })
        return output

    @staticmethod
    def _chunk_id(chunk: TextChunk) -> str:
        payload = "|".join([
            str(chunk.metadata.get("source_file", "")),
            str(chunk.metadata.get("page_number", "")),
            str(chunk.metadata.get("institute_code", "")),
            str(chunk.metadata.get("choice_code", "")),
            str(chunk.metadata.get("seat_allocation", "")),
            str(chunk.metadata.get("stage", "")),
            str(chunk.metadata.get("seat_cutoffs_json", "")),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None or value == "":
                continue
            clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        return clean
