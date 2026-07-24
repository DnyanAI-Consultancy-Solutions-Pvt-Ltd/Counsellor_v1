from __future__ import annotations

from pathlib import Path
import hashlib
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config.settings import get_settings
from rag.loaders import load_file
from rag.chunking import split_text
from rag.classifier import classify_document


class KnowledgeBase:
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="mhtcet_cap",
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model),
            metadata={"hnsw:space": "cosine"},
        )

    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict] = {}
        for metadata in data.get("metadatas", []):
            if not metadata:
                continue
            filename = metadata.get("filename", "unknown")
            grouped.setdefault(filename, {
                "filename": filename,
                "document_type": metadata.get("document_type", "unknown"),
                "chunks": 0,
                "size_bytes": metadata.get("size_bytes", 0),
            })
            grouped[filename]["chunks"] += 1
        return sorted(grouped.values(), key=lambda item: item["filename"].lower())

    def delete_document(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})

    def index_file(self, path: Path) -> dict:
        self.delete_document(path.name)
        pages = load_file(path)
        sample = " ".join(page["text"] for page in pages[:3])
        document_type = classify_document(path.name, sample)
        ids, documents, metadatas = [], [], []
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for page in pages:
            for chunk_number, chunk in enumerate(split_text(page["text"])):
                ids.append(f"{digest}-{page['page']}-{chunk_number}")
                documents.append(chunk)
                metadatas.append({
                    "filename": path.name,
                    "document_type": document_type,
                    "page": int(page["page"]),
                    "section": str(page.get("section", "")),
                    "size_bytes": int(path.stat().st_size),
                })
        if not documents:
            raise ValueError("No readable text found in document")
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return {
            "filename": path.name,
            "document_type": document_type,
            "chunks": len(documents),
            "size_bytes": path.stat().st_size,
        }

    def search_many(
        self,
        queries: list[str],
        *,
        document_types: list[str] | None = None,
        n_results_per_query: int | None = None,
        max_chunks: int | None = None,
    ) -> list[dict]:
        settings = get_settings()
        n_results = n_results_per_query or settings.retrieval_results_per_query
        maximum = max_chunks or settings.max_evidence_chunks
        available = self.list_documents()
        filenames = [
            item["filename"] for item in available
            if not document_types or item["document_type"] in document_types
        ]
        if not filenames:
            filenames = [item["filename"] for item in available]

        evidence: list[dict] = []
        for query in queries:
            for filename in filenames:
                result = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"filename": filename},
                    include=["documents", "metadatas", "distances"],
                )
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                for text, metadata, distance in zip(docs, metas, distances):
                    evidence.append({
                        "text": text,
                        "metadata": metadata,
                        "score": round(1 - float(distance), 4),
                    })

        seen: set[tuple] = set()
        unique: list[dict] = []
        for item in sorted(evidence, key=lambda value: value["score"], reverse=True):
            metadata = item["metadata"]
            key = (metadata.get("filename"), metadata.get("page"), item["text"][:160])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= maximum:
                break
        return unique
