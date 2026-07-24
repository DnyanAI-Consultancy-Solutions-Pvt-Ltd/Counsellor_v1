from typing import TypedDict, Any
from models.schemas import StudentProfile, Recommendation


class CounsellorState(TypedDict, total=False):
    session_id: str
    message: str
    profile: StudentProfile
    history: list[dict]
    existing_recommendations: list[Recommendation]
    workbook_version: int
    route: str
    answer: str
    recommendations: list[Recommendation]
    evidence: list[dict]
    confidence: str
    trace: list[str]
    handoff_reason: str
