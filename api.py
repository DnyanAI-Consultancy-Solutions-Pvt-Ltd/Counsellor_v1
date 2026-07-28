from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agents.counsellor import get_counsellor_agent
from agents.feedback import get_feedback_agent
from config.settings import get_settings
from models.schemas import CounsellingRequest, FeedbackRequest
from rag.store import ChromaStore
from services.excel_service import ExcelService
from services.session_service import SessionService

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MHT-CET Agentic RAG Counsellor V2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ChromaStore()
counsellor_agent = get_counsellor_agent()
feedback_agent = get_feedback_agent()
excel_service = ExcelService()
session_service = SessionService()
UPLOAD_DIRECTORY = Path("storage/uploads")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _save_upload(uploaded_file: UploadFile) -> Path:
    filename = Path(uploaded_file.filename or "").name.strip()
    if not filename:
        raise ValueError("Invalid filename.")
    destination = UPLOAD_DIRECTORY / filename
    try:
        with destination.open("wb") as target:
            shutil.copyfileobj(uploaded_file.file, target)
    finally:
        uploaded_file.file.close()
    return destination


def _detect_document_type(filename: str, supplied_type: str | None) -> str:
    if supplied_type:
        return supplied_type.strip().lower()
    name = filename.casefold()
    return "cutoff" if any(x in name for x in ("cutoff", "cut-off", "cap", "merit")) else "general"


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MHT-CET Agentic RAG Counsellor V2 is running.", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "collection": store.health(),
        "llm": counsellor_agent.llm.health(),
    }


@app.post("/upload")
def upload_documents(
    files: list[UploadFile] = File(...),
    document_type: str | None = Form(default=None),
) -> dict[str, Any]:
    results = []
    for uploaded_file in files:
        try:
            path = _save_upload(uploaded_file)
            result = store.index_document(
                path,
                document_type=_detect_document_type(path.name, document_type),
            )
            results.append({"status": "success", **result})
        except Exception as exc:
            logger.exception("Upload failed for %s", uploaded_file.filename)
            results.append({"status": "failed", "file_name": uploaded_file.filename, "error": str(exc)})

    if not results or all(item["status"] == "failed" for item in results):
        raise HTTPException(status_code=500, detail=results or "No files uploaded.")

    return {"status": "success", "collection_chunks": store.document_count, "results": results}


@app.post("/counsel")
def counsel(request: CounsellingRequest) -> dict[str, Any]:
    try:
        result = counsellor_agent.counsel(
            request.student_profile.model_dump(exclude_none=True),
            request.user_request,
        )
        session_id = session_service.create(result)
        workbook = excel_service.export_recommendations(
            result.get("recommendations", []),
            session_service.workbook_path(session_id),
        )
        return {**result, "session_id": session_id, "excel_download_url": f"/download/{workbook.name}"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Counselling failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    try:
        previous = session_service.load(request.session_id)
        updated = feedback_agent.apply(previous, request.feedback)
        session_service.save(request.session_id, updated)
        workbook = excel_service.export_recommendations(
            updated.get("recommendations", []),
            session_service.workbook_path(request.session_id),
        )
        return {**updated, "session_id": request.session_id, "excel_download_url": f"/download/{workbook.name}"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feedback failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = Path("storage/outputs") / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=path,
        filename=safe_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.delete("/knowledge-base")
def reset_knowledge_base() -> dict[str, Any]:
    store.reset_collection()
    return {"status": "success", "message": "Knowledge base reset.", "collection": store.health()}
