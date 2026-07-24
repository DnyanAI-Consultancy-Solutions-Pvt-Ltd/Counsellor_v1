from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class StudentProfile(BaseModel):
    mht_cet_percentile: float | None = Field(default=None, ge=0, le=100)
    jee_percentile: float | None = Field(default=None, ge=0, le=100)
    category: str | None = None
    gender: str | None = None
    home_university: str | None = None
    preferred_branches: list[str] = Field(default_factory=list)
    preferred_cities: list[str] = Field(default_factory=list)
    institute_preferences: list[str] = Field(default_factory=list)
    minority_status: str | None = None
    tfws: bool | None = None
    pwd: bool | None = None


class Recommendation(BaseModel):
    institute_code: str = ""
    college: str
    city: str = ""
    course_code: str = ""
    course: str
    seat_type: str
    cutoff_percentile: float | None = Field(default=None, ge=0, le=100)
    match_category: Literal["Dream", "Target", "Safe"]
    reasoning_logic: str
    source_filename: str = ""
    source_page: int | None = None

    @property
    def row_key(self) -> str:
        parts = [self.institute_code, self.college, self.course_code, self.course, self.seat_type]
        return "|".join(str(part).strip().lower() for part in parts)


class AgentRoute(BaseModel):
    action: Literal["generate_initial", "handoff_to_feedback", "clarify", "answer_only"]
    updated_profile: StudentProfile
    clarification_question: str = ""
    response: str = ""
    reasoning: str = ""


class RetrievalPlan(BaseModel):
    search_queries: list[str] = Field(min_length=2, max_length=12)
    document_types: list[str] = Field(default_factory=lambda: ["cutoff"])
    rationale: str = ""


class RecommendationBatch(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)
    counselling_summary: str
    assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["High", "Medium", "Low"] = "Medium"


class FeedbackChangeSet(BaseModel):
    remove_keys: list[str] = Field(default_factory=list)
    add_or_update: list[Recommendation] = Field(default_factory=list)
    preferred_order: list[str] = Field(default_factory=list)
    sort_mode: Literal["keep_existing", "cutoff_desc", "cutoff_asc", "preferred_order"] = "keep_existing"
    requires_additional_retrieval: bool = False
    retrieval_queries: list[str] = Field(default_factory=list, max_length=8)
    explanation: str
    confidence: Literal["High", "Medium", "Low"] = "Medium"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=8000)
    profile: StudentProfile | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    profile: StudentProfile
    recommendations: list[Recommendation] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    workbook_available: bool = False
    workbook_version: int = 0
    route: Literal["Counsellor_Agent", "Feedback_Agent"]
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    confidence: Literal["High", "Medium", "Low"] = "Low"


class DocumentInfo(BaseModel):
    filename: str
    document_type: str
    chunks: int
    size_bytes: int
