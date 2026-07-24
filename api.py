import logging
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import RateLimitError

from config.settings import get_settings
from graph.workflow import workflow
from memory.store import SessionStore
from models.schemas import ChatRequest, ChatResponse, StudentProfile, DocumentInfo, Recommendation
from services.knowledge_service import KnowledgeService
from services.recommendation_service import counts
from services.workbook_service import WorkbookService

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store = SessionStore()
knowledge = KnowledgeService()
workbooks = WorkbookService()


@app.get("/")
def root():
    return {"message": settings.app_name, "status": "running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "application": settings.app_name,
        "environment": settings.app_env,
        "documents": len(knowledge.kb.list_documents()),
        "agents": ["Counsellor_Agent", "Feedback_Agent"],
        "minimum_recommendations": settings.minimum_recommendations,
    }


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return knowledge.kb.list_documents()


def check_admin(key: str | None):
    if not key or key != settings.admin_key:
        raise HTTPException(401, "Invalid admin key")


@app.post("/documents/upload", response_model=DocumentInfo)
def upload(file: UploadFile = File(...), x_admin_key: str | None = Header(default=None)):
    check_admin(x_admin_key)
    try:
        return knowledge.save_and_index(file)
    except Exception as exc:
        log.exception("Indexing failed")
        raise HTTPException(400, str(exc)) from exc


@app.delete("/documents/{filename}")
def delete(filename: str, x_admin_key: str | None = Header(default=None)):
    check_admin(x_admin_key)
    knowledge.kb.delete_document(filename)
    return {"deleted": filename}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    store.clear(session_id)
    session_dir = settings.sessions_dir / session_id
    if session_dir.exists():
        for path in session_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
        session_dir.rmdir()
    return {"cleared": session_id}


@app.get("/sessions/{session_id}/workbook")
def download_workbook(session_id: str):
    path = workbooks.workbook_path(session_id)
    if not path.exists():
        raise HTTPException(404, "No preference workbook exists for this session")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="cutoff_list.xlsx",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        existing = store.get(payload.session_id)
        saved = StudentProfile.model_validate(existing.get("profile", {}))
        incoming = payload.profile or StudentProfile()
        merged = StudentProfile.model_validate({
            **saved.model_dump(),
            **{key: value for key, value in incoming.model_dump().items() if value not in (None, [], "")},
        })
        existing_rows = [Recommendation.model_validate(row) for row in existing.get("recommendations", [])]

        result = workflow.invoke({
            "session_id": payload.session_id,
            "message": payload.message,
            "profile": merged,
            "history": existing.get("history", []),
            "existing_recommendations": existing_rows,
            "workbook_version": existing.get("workbook_version", 0),
            "trace": [],
        })

        answer = result.get("answer", "")
        profile = result.get("profile", merged)
        recommendations = result.get("recommendations", existing_rows)
        route = result.get("route", "Counsellor_Agent")
        workbook_version = int(existing.get("workbook_version", 0))
        workbook_available = bool(recommendations)
        if workbook_available:
            workbooks.save_atomic(payload.session_id, recommendations)
            workbook_version += 1

        store.save(
            payload.session_id,
            profile,
            payload.message,
            answer,
            recommendations,
            workbook_version,
        )

        evidence = result.get("evidence", [])
        sources = []
        seen = set()
        for item in evidence:
            metadata = item.get("metadata", {})
            key = (metadata.get("filename"), metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "filename": key[0],
                "page": key[1],
                "document_type": metadata.get("document_type"),
                "score": item.get("score"),
            })

        return ChatResponse(
            session_id=payload.session_id,
            answer=answer,
            profile=profile,
            recommendations=recommendations,
            counts=counts(recommendations),
            workbook_available=workbook_available,
            workbook_version=workbook_version,
            route=route,
            sources=sources,
            trace=result.get("trace", []),
            confidence=result.get("confidence", "Low"),
        )
    except RateLimitError as exc:
        message = str(exc)
        match = re.search(r"try again in\s+([^.;]+)", message, re.IGNORECASE)
        retry = match.group(1).strip() if match else "a few minutes"
        raise HTTPException(429, f"Groq rate limit reached. Retry in {retry}.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Chat failed")
        raise HTTPException(500, f"Counsellor failed: {exc}") from exc
