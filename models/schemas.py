from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    percentile: float = Field(ge=0, le=100)
    category: str = "OPEN"
    gender: str | None = None
    preferred_branch: str | None = None
    preferred_branches: list[str] = Field(default_factory=list)
    preferred_location: str | None = None
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_university: str | None = None
    college_preference: str | None = None
    home_university: str | None = None
    seat_type: str | None = None
    college_count: int = Field(default=30, ge=1, le=100)
    additional_preferences: dict[str, Any] = Field(default_factory=dict)


class CounsellingRequest(BaseModel):
    student_profile: StudentProfile
    user_request: str | None = (
        "Generate a ranked CAP preference list with Dream, Target and Safe options."
    )


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str
