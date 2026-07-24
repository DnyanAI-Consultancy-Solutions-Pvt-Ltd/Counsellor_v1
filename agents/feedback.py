from __future__ import annotations

import json

from agents.prompts import FEEDBACK_PROMPT, FEEDBACK_FINAL_PROMPT
from models.llm import get_llm
from models.schemas import FeedbackChangeSet, Recommendation, StudentProfile
from rag.store import KnowledgeBase
from services.recommendation_service import apply_change_set


def _rows_payload(rows: list[Recommendation]) -> list[dict]:
    return [{**row.model_dump(), "row_key": row.row_key} for row in rows]


def _evidence_payload(evidence: list[dict]) -> list[dict]:
    return [
        {
            "source_filename": item["metadata"].get("filename", ""),
            "source_page": item["metadata"].get("page"),
            "text": item["text"][:1800],
            "score": item.get("score"),
        }
        for item in evidence
    ]


def apply_feedback(
    message: str,
    profile: StudentProfile,
    history: list[dict],
    existing: list[Recommendation],
) -> tuple[list[Recommendation], str, str, list[dict], list[str]]:
    model = get_llm(max_tokens=8000, temperature=0.0).with_structured_output(FeedbackChangeSet)
    changes = model.invoke(f"""{FEEDBACK_PROMPT}

PROFILE:
{profile.model_dump_json()}

EXISTING PREFERENCE SHEET:
{json.dumps(_rows_payload(existing), ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-6:], ensure_ascii=False)}

STUDENT FEEDBACK:
{message}
""")

    evidence: list[dict] = []
    if changes.requires_additional_retrieval:
        queries = changes.retrieval_queries or [message]
        evidence = KnowledgeBase().search_many(queries, document_types=["cutoff"])
        changes = model.invoke(f"""{FEEDBACK_FINAL_PROMPT}

PROFILE:
{profile.model_dump_json()}

EXISTING PREFERENCE SHEET:
{json.dumps(_rows_payload(existing), ensure_ascii=False)}

STUDENT FEEDBACK:
{message}

RETRIEVED CUTOFF EVIDENCE:
{json.dumps(_evidence_payload(evidence), ensure_ascii=False)}
""")

    updated = apply_change_set(existing, changes)
    trace = [
        "Counsellor Agent handed the existing preference sheet to Feedback Agent",
        "Feedback Agent interpreted the requested in-place modification",
    ]
    if evidence:
        trace.append(f"Feedback Agent retrieved {len(evidence)} additional cutoff chunks")
    trace.append(f"Feedback Agent updated the workbook from {len(existing)} to {len(updated)} rows")
    return updated, changes.explanation, changes.confidence, evidence, trace
