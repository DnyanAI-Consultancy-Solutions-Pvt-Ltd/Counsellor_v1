from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.counsellor import get_counsellor_agent
from config.settings import get_settings
from rag.store import ChromaStore


# ==========================================================
# Logging
# ==========================================================

settings = get_settings()

logging.basicConfig(
    level=getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ==========================================================
# Application
# ==========================================================

app = FastAPI(
    title="MHT-CET Agentic RAG Counsellor",
    version="4.0.0",
    description=(
        "Agentic RAG counselling backend for "
        "MHT-CET CAP engineering admissions."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Shared services
# ==========================================================

store = ChromaStore()
counsellor_agent = get_counsellor_agent()

UPLOAD_DIRECTORY = Path("storage/uploads")

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# API schemas
# ==========================================================

class StudentProfileRequest(BaseModel):
    percentile: float = Field(
        ge=0,
        le=100,
    )

    category: str = "OPEN"

    gender: str | None = None

    preferred_branch: str | None = None

    preferred_branches: list[str] = Field(
        default_factory=list,
    )

    preferred_location: str | None = None

    preferred_locations: list[str] = Field(
        default_factory=list,
    )

    college_preference: str | None = None

    home_university: str | None = None

    seat_type: str | None = None


class CounsellingRequest(BaseModel):
    student_profile: StudentProfileRequest

    user_request: str | None = (
        "Generate a ranked CAP preference list "
        "with Dream, Target and Safe options."
    )


class HealthResponse(BaseModel):
    status: str
    application: str
    collection: dict[str, Any]
    llm: dict[str, Any]


# ==========================================================
# Utility functions
# ==========================================================

def _safe_filename(
    filename: str,
) -> str:
    """
    Prevent directory traversal and unsafe file names.
    """

    clean_name = Path(filename).name.strip()

    if not clean_name:
        raise ValueError(
            "Uploaded file has no valid filename."
        )

    return clean_name


def _detect_document_type(
    filename: str,
    supplied_type: str | None,
) -> str:
    """
    Use the supplied type when present.

    Otherwise classify likely CAP cutoff documents by name.
    This only affects metadata filtering, not counselling
    decisions.
    """

    if supplied_type:
        return supplied_type.strip().lower()

    lower_name = filename.lower()

    cutoff_keywords = (
        "cutoff",
        "cut-off",
        "cap",
        "merit",
        "allotment",
        "institute",
    )

    if any(
        keyword in lower_name
        for keyword in cutoff_keywords
    ):
        return "cutoff"

    return "general"


def _save_upload(
    uploaded_file: UploadFile,
) -> Path:
    """
    Save one uploaded file to local storage.
    """

    filename = _safe_filename(
        uploaded_file.filename or ""
    )

    destination = (
        UPLOAD_DIRECTORY / filename
    )

    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(
                uploaded_file.file,
                output_file,
            )
    finally:
        uploaded_file.file.close()

    return destination


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": (
            "MHT-CET Agentic RAG Counsellor "
            "backend is running."
        ),
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> dict[str, Any]:
    """
    Check backend, ChromaDB and LLM configuration.
    """

    return {
        "status": "healthy",
        "application": (
            "MHT-CET Agentic RAG Counsellor"
        ),
        "collection": store.health(),
        "llm": counsellor_agent.llm.health(),
    }


@app.post("/upload")
def upload_documents(
    files: list[UploadFile] = File(...),
    document_type: str | None = Form(
        default=None
    ),
) -> dict[str, Any]:
    """
    Upload and index multiple documents.

    Supported formats are controlled by DocumentLoader,
    currently PDF, DOCX, TXT and Markdown.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded.",
        )

    results: list[dict[str, Any]] = []
    total_indexed = 0
    total_duplicates = 0
    failed_files = 0

    for uploaded_file in files:
        original_name = (
            uploaded_file.filename
            or "unnamed-file"
        )

        try:
            saved_path = _save_upload(
                uploaded_file
            )

            resolved_document_type = (
                _detect_document_type(
                    filename=saved_path.name,
                    supplied_type=document_type,
                )
            )

            ingestion_result = (
                store.index_document(
                    file_path=saved_path,
                    document_type=(
                        resolved_document_type
                    ),
                )
            )

            total_indexed += int(
                ingestion_result.get(
                    "indexed",
                    0,
                )
            )

            total_duplicates += int(
                ingestion_result.get(
                    "duplicates",
                    0,
                )
            )

            results.append(
                {
                    "status": "success",
                    **ingestion_result,
                }
            )

        except Exception as exc:
            failed_files += 1

            logger.exception(
                "Failed to index file: %s",
                original_name,
            )

            results.append(
                {
                    "status": "failed",
                    "file_name": original_name,
                    "error": str(exc),
                }
            )

    if failed_files == len(files):
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "All uploaded files failed "
                    "during ingestion."
                ),
                "results": results,
            },
        )

    return {
        "status": (
            "partial_success"
            if failed_files
            else "success"
        ),
        "files_received": len(files),
        "files_failed": failed_files,
        "total_chunks_indexed": total_indexed,
        "total_duplicates": total_duplicates,
        "collection_chunks": (
            store.document_count
        ),
        "results": results,
    }


@app.post("/counsel")
def counsel(
    request: CounsellingRequest,
) -> dict[str, Any]:
    """
    Run the agentic RAG counselling workflow.
    """

    try:
        profile = (
            request.student_profile.model_dump(
                exclude_none=True,
            )
        )

        result = counsellor_agent.counsel(
            student_profile=profile,
            user_request=request.user_request,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Counsellor workflow failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Counsellor failed: {exc}"
            ),
        ) from exc


@app.delete("/knowledge-base")
def reset_knowledge_base() -> dict[str, Any]:
    """
    Delete all indexed chunks.

    Keep this endpoint for development and demo resets.
    """

    try:
        store.reset_collection()

        return {
            "status": "success",
            "message": (
                "Knowledge base was reset."
            ),
            "collection": store.health(),
        }

    except Exception as exc:
        logger.exception(
            "Knowledge-base reset failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to reset knowledge base: "
                f"{exc}"
            ),
        ) from exc