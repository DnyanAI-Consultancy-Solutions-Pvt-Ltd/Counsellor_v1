from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config.settings import get_settings
from rag.chunker import DocumentChunker, TextChunk
from rag.embeddings import get_embedding_service
from rag.loader import DocumentLoader
from services.university_mapping import UniversityMappingService


class ChromaStore:
    """
    Persistent ChromaDB store.

    Supports:
    - Collection creation
    - Document loading and chunking
    - Embedding generation
    - Duplicate-safe indexing
    - Semantic search
    - Metadata filtering
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        self.embedding_service = get_embedding_service()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.university_mapping = UniversityMappingService()

        Path(self.settings.chroma_db_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.settings.chroma_db_path,
        )

        self.collection = self._load_collection()

    # =======================================================
    # Collection management
    # =======================================================

    def _load_collection(self) -> Collection:
        """
        Create or load the configured collection.
        """

        return self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={
                "description": (
                    "MHT-CET CAP Counselling Knowledge Base"
                ),
                "embedding_model": (
                    self.settings.embedding_model
                ),
            },
        )

    @property
    def collection_name(self) -> str:
        return self.collection.name

    @property
    def document_count(self) -> int:
        """
        Number of indexed chunks.
        """

        return self.collection.count()

    @property
    def embedding_dimension(self) -> int:
        return (
            self.embedding_service.embedding_dimension
        )

    @property
    def storage_path(self) -> str:
        return self.settings.chroma_db_path

    def collection_exists(self) -> bool:
        """
        Check whether the configured collection exists.
        """

        collections = self.client.list_collections()

        collection_names = [
            collection.name
            for collection in collections
        ]

        return (
            self.settings.collection_name
            in collection_names
        )

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.
        """

        if self.collection_exists():
            self.client.delete_collection(
                name=self.settings.collection_name,
            )

        self.collection = self._load_collection()

    def health(self) -> dict[str, Any]:
        """
        Return collection health details.
        """

        return {
            "collection": self.collection_name,
            "storage_path": self.storage_path,
            "chunks": self.document_count,
            "embedding_model": (
                self.settings.embedding_model
            ),
            "embedding_dimension": (
                self.embedding_dimension
            ),
        }

    # =======================================================
    # Document indexing
    # =======================================================

    def index_document(
        self,
        file_path: str | Path,
        document_type: str = "general",
    ) -> dict[str, Any]:
        """
        Load, classify, chunk, embed and index one document.

        Safe auto-detection behaviour:
        - Existing CAP cutoff parsing is tried first when the caller sends
          ``document_type="cutoff"``.
        - If at least one structured CAP cutoff record is produced, the document
          remains a cutoff document and the existing counselling flow is unchanged.
        - If no structured CAP cutoff record is produced, the document is re-chunked
          as ``general`` so supporting documents (university directories, brochures,
          notices, etc.) can be retrieved separately.

        This keeps the current CAP functionality intact while allowing mixed
        knowledge-base uploads without requiring the UI to classify every file.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        pages = self.loader.load(path)

        requested_type = str(document_type or "general").strip().casefold()
        if requested_type not in {"cutoff", "general", "auto"}:
            requested_type = "general"

        detected_type = requested_type
        auto_detected = False

        # Preserve the existing behaviour for explicitly general documents.
        if requested_type == "general":
            chunks = self.chunker.chunk_pages(
                pages=pages,
                source_file=path.name,
                document_type="general",
            )

        else:
            # Existing CAP parser first. This is the backwards-compatible path.
            cutoff_chunks = self.chunker.chunk_pages(
                pages=pages,
                source_file=path.name,
                document_type="cutoff",
            )

            structured_cutoff_count = sum(
                1
                for chunk in cutoff_chunks
                if str(chunk.metadata.get("record_type", "")).casefold()
                == "cap_branch_cutoff"
            )

            if structured_cutoff_count > 0:
                detected_type = "cutoff"
                chunks = cutoff_chunks
            else:
                # No usable CAP records were produced. Treat the same source as a
                # supporting/general document instead of polluting the cutoff pool.
                detected_type = "general"
                auto_detected = True
                chunks = self.chunker.chunk_pages(
                    pages=pages,
                    source_file=path.name,
                    document_type="general",
                )

        chunks = self._enrich_cutoff_chunks_with_university(chunks)
        result = self.index_chunks(chunks)

        structured_cutoff_records = sum(
            1
            for chunk in chunks
            if str(chunk.metadata.get("record_type", "")).casefold()
            == "cap_branch_cutoff"
        )
        generic_chunks = sum(
            1
            for chunk in chunks
            if str(chunk.metadata.get("record_type", "")).casefold()
            == "generic_text"
        )

        return {
            "file_name": path.name,
            "requested_document_type": requested_type,
            "document_type": detected_type,
            "auto_detected": auto_detected,
            "pages_loaded": len(pages),
            "chunks_created": len(chunks),
            "structured_cutoff_records": structured_cutoff_records,
            "generic_chunks": generic_chunks,
            **result,
        }

    def _enrich_cutoff_chunks_with_university(
        self,
        chunks: list[TextChunk],
    ) -> list[TextChunk]:
        """Enrich structured CAP records with university using Institute Code."""
        code_to_university: dict[str, str] = {}

        for university in self.university_mapping.list_universities():
            resolved = self.university_mapping.resolve(university)
            if resolved.get("status") != "resolved":
                continue

            matched_university = str(
                resolved.get("matched_university") or university
            ).strip()

            for code in resolved.get("institute_codes") or []:
                normalized = self._normalise_institute_code(code)
                if normalized:
                    code_to_university[normalized] = matched_university

        enriched: list[TextChunk] = []

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})

            if str(metadata.get("record_type", "")).strip().casefold() == "cap_branch_cutoff":
                institute_code = self._normalise_institute_code(
                    metadata.get("institute_code")
                    or metadata.get("institute")
                    or metadata.get("college_code")
                )
                university = code_to_university.get(institute_code)

                if university:
                    metadata["university"] = university
                    metadata["university_match_method"] = "institute_code"
                else:
                    metadata["university"] = ""
                    metadata["university_match_method"] = "unmapped"

            enriched.append(TextChunk(text=chunk.text, metadata=metadata))

        return enriched

    @staticmethod
    def _normalise_institute_code(value: Any) -> str:
        text = str(value or "").strip()
        if text.endswith(".0"):
            text = text[:-2]
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def index_chunks(
        self,
        chunks: list[TextChunk],
    ) -> dict[str, int]:
        """
        Index non-duplicate chunks.
        """

        if not chunks:
            return {
                "indexed": 0,
                "duplicates": 0,
            }

        documents: list[str] = []
        chunk_ids: list[str] = []
        metadatas: list[
            dict[str, str | int | float | bool]
        ] = []

        duplicate_count = 0

        for chunk in chunks:
            chunk_id = self._chunk_id(chunk)

            if self.chunk_exists(chunk_id):
                duplicate_count += 1
                continue

            clean_text = chunk.text.strip()

            if not clean_text:
                continue

            documents.append(clean_text)
            chunk_ids.append(chunk_id)
            metadatas.append(
                self._prepare_metadata(
                    chunk.metadata
                )
            )

        if not documents:
            return {
                "indexed": 0,
                "duplicates": duplicate_count,
            }

        embeddings = (
            self.embedding_service.embed_documents(
                documents
            )
        )

        # ChromaDB enforces a maximum number of records per add() call.
        # Use a conservative fixed batch size so large CAP documents can be
        # indexed safely across ChromaDB versions/configurations.
        batch_size = 500
        indexed_count = 0

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))

            self.collection.add(
                ids=chunk_ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

            indexed_count += end - start

            print(
                f"Chroma indexing batch: "
                f"{start + 1}-{end} / {len(documents)}"
            )

        return {
            "indexed": indexed_count,
            "duplicates": duplicate_count,
        }

    # =======================================================
    # Duplicate detection
    # =======================================================

    def chunk_exists(
        self,
        chunk_id: str,
    ) -> bool:
        """
        Check whether a chunk already exists.
        """

        result = self.collection.get(
            ids=[chunk_id],
            include=[],
        )

        ids = result.get("ids") or []

        return len(ids) > 0

    @staticmethod
    def _chunk_id(
        chunk: TextChunk,
    ) -> str:
        """
        Generate a stable SHA-256 chunk ID.
        """

        source_file = str(
            chunk.metadata.get(
                "source_file",
                "",
            )
        )

        page_number = str(
            chunk.metadata.get(
                "page_number",
                "",
            )
        )

        document_type = str(
            chunk.metadata.get(
                "document_type",
                "general",
            )
        )

        chunk_index = str(
            chunk.metadata.get(
                "chunk_index",
                "",
            )
        )

        raw_value = "|".join(
            [
                source_file,
                page_number,
                document_type,
                chunk_index,
                chunk.text.strip(),
            ]
        )

        return hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _prepare_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """
        Convert metadata to Chroma-supported scalar values.
        """

        prepared: dict[
            str,
            str | int | float | bool
        ] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                prepared[str(key)] = value
            else:
                prepared[str(key)] = str(value)

        return prepared

    # =======================================================
    # Semantic search
    # =======================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_type: str | None = None,
        source_file: str | None = None,
        university: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search with optional metadata filters.
        """

        clean_query = query.strip()

        if not clean_query:
            return []

        if self.document_count == 0:
            return []

        result_limit = (
            top_k
            or self.settings.results_per_query
        )

        result_limit = max(
            1,
            min(
                result_limit,
                self.document_count,
            ),
        )

        query_embedding = (
            self.embedding_service.embed_query(
                clean_query
            )
        )

        where_filter = self._build_where_filter(
            document_type=document_type,
            source_file=source_file,
            university=university,
        )

        query_arguments: dict[str, Any] = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": result_limit,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter is not None:
            query_arguments["where"] = (
                where_filter
            )

        raw_results = self.collection.query(
            **query_arguments
        )

        return self._format_search_results(
            raw_results
        )

    @staticmethod
    def _build_where_filter(
        document_type: str | None = None,
        source_file: str | None = None,
        university: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Build Chroma metadata filters.
        """

        filters: list[dict[str, Any]] = []

        if document_type:
            filters.append(
                {
                    "document_type": {
                        "$eq": document_type
                    }
                }
            )

        if source_file:
            filters.append(
                {
                    "source_file": {
                        "$eq": source_file
                    }
                }
            )

        if university:
            filters.append(
                {"university": {"$eq": university}}
            )

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {
            "$and": filters
        }

    @staticmethod
    def _format_search_results(
        raw_results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Convert Chroma results into a clean list.
        """

        documents_groups = (
            raw_results.get("documents")
            or [[]]
        )

        metadata_groups = (
            raw_results.get("metadatas")
            or [[]]
        )

        distance_groups = (
            raw_results.get("distances")
            or [[]]
        )

        documents = (
            documents_groups[0]
            if documents_groups
            else []
        )

        metadatas = (
            metadata_groups[0]
            if metadata_groups
            else []
        )

        distances = (
            distance_groups[0]
            if distance_groups
            else []
        )

        formatted_results: list[
            dict[str, Any]
        ] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            distance_value = float(distance)

            similarity = max(
                0.0,
                min(
                    1.0,
                    1.0 - distance_value,
                ),
            )

            formatted_results.append(
                {
                    "text": document or "",
                    "metadata": metadata or {},
                    "distance": round(
                        distance_value,
                        6,
                    ),
                    "similarity": round(
                        similarity,
                        6,
                    ),
                }
            )

        formatted_results.sort(
            key=lambda item: item[
                "similarity"
            ],
            reverse=True,
        )

        return formatted_results

    # =======================================================
    # Useful wrappers
    # =======================================================

    def search_cutoffs(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search only cutoff documents.
        """

        return self.search(
            query=query,
            top_k=top_k,
            document_type="cutoff",
        )

    def list_cutoff_records(
        self,
        document_type: str = "cutoff",
        university: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all structured cutoff records stored in ChromaDB.

        This is used by Retriever.retrieve_cutoff_pool() to make sure
        counselling considers the complete indexed CAP cutoff pool instead
        of relying only on semantic top-k retrieval.
        """
        if self.document_count == 0:
            return []

        filters: list[dict[str, Any]] = []
        if document_type:
            filters.append({"document_type": {"$eq": document_type}})
        if university:
            filters.append({"university": {"$eq": university}})

        if not filters:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}

        output: list[dict[str, Any]] = []

        # Keep each Chroma read comfortably below large collection limits.
        page_size = 1000
        offset = 0

        while True:
            kwargs: dict[str, Any] = {
                "include": [
                    "documents",
                    "metadatas",
                ],
                "limit": page_size,
                "offset": offset,
            }

            if where is not None:
                kwargs["where"] = where

            raw = self.collection.get(**kwargs)

            ids = raw.get("ids") or []
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []

            if not ids:
                break

            for chunk_id, document, metadata in zip(
                ids,
                documents,
                metadatas,
            ):
                metadata = metadata or {}

                # Only structured CAP branch/cutoff records should enter
                # deterministic counselling.
                if (
                    str(
                        metadata.get(
                            "record_type",
                            ""
                        )
                    ).strip().casefold()
                    != "cap_branch_cutoff"
                ):
                    continue

                output.append(
                    {
                        "id": chunk_id,
                        "text": document or "",
                        "metadata": metadata,
                        "distance": 1.0,
                        "similarity": 0.0,
                    }
                )

            offset += len(ids)

            if len(ids) < page_size:
                break

        return output

    def search_by_source(
        self,
        query: str,
        source_file: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search only one source file.
        """

        return self.search(
            query=query,
            top_k=top_k,
            source_file=source_file,
        )

    def __repr__(self) -> str:
        return (
            "ChromaStore("
            f"collection='{self.collection_name}', "
            f"chunks={self.document_count}"
            ")"
        )