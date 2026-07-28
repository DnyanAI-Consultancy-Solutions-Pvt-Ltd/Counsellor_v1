from __future__ import annotations

from typing import Any

from agents.counsellor import CounsellorAgent, get_counsellor_agent
from services.llm_service import LLMService, get_llm_service
from services.preference_service import PreferenceService


class FeedbackAgent:
    """Interpret feedback agentically, execute it deterministically, and validate it."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        counsellor_agent: CounsellorAgent | None = None,
        preference_service: PreferenceService | None = None,
    ) -> None:
        self.llm = llm_service or get_llm_service()
        self.counsellor = counsellor_agent or get_counsellor_agent()
        self.preferences = preference_service or PreferenceService()

    def apply(self, previous_result: dict[str, Any], feedback: str) -> dict[str, Any]:
        if not feedback.strip():
            raise ValueError("Feedback cannot be empty.")

        current = self.preferences.normalise(previous_result.get("recommendations", []))
        plan = self._plan(previous_result, feedback)
        notes: list[str] = []
        audit: list[dict[str, Any]] = []

        for operation in plan.get("operations", []):
            if not isinstance(operation, dict):
                continue
            action = str(operation.get("action", "")).strip().casefold()
            field = operation.get("field")
            value = str(operation.get("value", "")).strip()
            filters = operation.get("filters") if isinstance(operation.get("filters"), dict) else {}

            if action == "remove":
                before = len(current)
                current, removed = self.preferences.remove_matching(current, field=field, value=value)
                remaining = self.preferences.count_matching(current, field=field, value=value)
                audit.append({
                    "action": "remove", "requested_value": value,
                    "removed_count": len(removed), "remaining_matches": remaining,
                    "before_count": before, "after_count": len(current),
                })
                if remaining:
                    raise RuntimeError(f"Removal validation failed: {remaining} matching recommendations remain for '{value}'.")
                notes.append(f"Removed {len(removed)} recommendation(s) matching {value or field}.")

            elif action == "add":
                requested = max(1, int(operation.get("count", 1) or 1))
                candidate_request = self._addition_request(feedback, operation)
                generated = self.counsellor.counsel(
                    previous_result.get("student_profile", {}),
                    candidate_request,
                )
                candidates = [
                    item for item in generated.get("recommendations", [])
                    if self.preferences.matches_filters(item, filters)
                ]
                before = len(current)
                current, added = self.preferences.add_exact(current, candidates, requested)
                audit.append({
                    "action": "add", "requested_count": requested,
                    "added_count": len(added), "before_count": before,
                    "after_count": len(current), "filters": filters,
                })
                if len(added) != requested:
                    notes.append(
                        f"Requested {requested} addition(s), but only {len(added)} unique grounded option(s) "
                        "were available in the retrieved CAP evidence."
                    )
                else:
                    notes.append(f"Added exactly {requested} unique grounded recommendation(s).")

            elif action in {"rerank", "prioritize", "reorder"}:
                direction = str(operation.get("direction", "top"))
                current = self.preferences.rerank(current, field=field, value=value, direction=direction)
                audit.append({"action": "rerank", "field": field, "value": value, "direction": direction})
                notes.append(f"Re-ranked recommendations with {value or field} toward the {direction}.")

            elif action == "replace":
                # Replace is executed as remove followed by an exact add.
                current, removed = self.preferences.remove_matching(current, field=field, value=value)
                requested = max(1, int(operation.get("count", len(removed) or 1) or 1))
                generated = self.counsellor.counsel(
                    previous_result.get("student_profile", {}),
                    self._addition_request(feedback, operation),
                )
                candidates = [
                    item for item in generated.get("recommendations", [])
                    if self.preferences.matches_filters(item, filters)
                ]
                current, added = self.preferences.add_exact(current, candidates, requested)
                audit.append({"action": "replace", "removed_count": len(removed), "requested_add_count": requested, "added_count": len(added)})
                notes.append(f"Replaced {len(removed)} recommendation(s) and added {len(added)} grounded alternative(s).")

        updated = dict(previous_result)
        updated["recommendations"] = self.preferences.resequence(current)
        updated["feedback"] = feedback
        updated["feedback_plan"] = plan
        updated["feedback_notes"] = notes or ["No executable list change was identified in the feedback."]
        updated["feedback_audit"] = audit
        requested_counts = previous_result.get("requested_zone_counts")
        validation = self.preferences.validate(current, requested_counts if isinstance(requested_counts, dict) else None)
        updated["validation"] = validation
        updated["zone_counts"] = validation["zone_counts"]
        updated["status"] = "success" if validation["is_valid"] else "partial_success"
        updated["summary"] = self._summary(updated, notes)
        return updated

    def _plan(self, previous_result: dict[str, Any], feedback: str) -> dict[str, Any]:
        return self.llm.generate_json(
            system_prompt="""
You are the feedback-planning agent for an MHT-CET preference list.
Convert the student's message into precise list operations. Do not rewrite the list.
Supported actions: remove, add, rerank, replace.

Rules:
- For "add 2 colleges", return action=add and count=2.
- For "remove Pune colleges", return action=remove, field=location, value=Pune.
- For branch or college requests, use field=branch or field=college.
- Put requested constraints for additions in filters, such as location, branch, zone, or college.
- Keep operations in the order the student requested.
- Do not invent list rows.

Return:
{
  "intent": "modify_list" | "question_only",
  "operations": [
    {
      "action": "remove|add|rerank|replace",
      "count": 1,
      "field": "location|college|branch|zone|seat_type",
      "value": "...",
      "direction": "top|bottom",
      "filters": {"location": "...", "branch": "...", "zone": "...", "college": "..."}
    }
  ],
  "rationale": "..."
}
""",
            user_prompt=(
                f"CURRENT LIST COUNT: {len(previous_result.get('recommendations', []))}\n"
                f"STUDENT PROFILE: {previous_result.get('student_profile', {})}\n"
                f"FEEDBACK: {feedback}"
            ),
            temperature=0.0,
            max_tokens=1600,
        )

    @staticmethod
    def _addition_request(feedback: str, operation: dict[str, Any]) -> str:
        count = max(1, int(operation.get("count", 1) or 1))
        filters = operation.get("filters") if isinstance(operation.get("filters"), dict) else {}
        return (
            f"Generate additional grounded alternatives for this feedback: {feedback}. "
            f"Need at least {count} unique new college-and-branch choices. "
            f"Apply these requested filters: {filters}. Prefer diversity and do not repeat existing choices."
        )

    @staticmethod
    def _summary(updated: dict[str, Any], notes: list[str]) -> str:
        return " ".join(notes) if notes else str(updated.get("summary", "Preference list reviewed."))


_feedback_agent: FeedbackAgent | None = None


def get_feedback_agent() -> FeedbackAgent:
    global _feedback_agent
    if _feedback_agent is None:
        _feedback_agent = FeedbackAgent()
    return _feedback_agent
