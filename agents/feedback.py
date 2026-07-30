from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agents.counsellor import CounsellorAgent, get_counsellor_agent
from services.llm_service import LLMService, get_llm_service
from services.preference_service import PreferenceService


class FeedbackAgent:
    """Agentic feedback counsellor with grounded, deterministic execution.

    The LLM decides the counselling action and explains the trade-offs. Python
    retrieves evidence, validates candidate IDs, prevents hallucinated records,
    checks duplicates, and safely mutates the list.
    """

    SUPPORTED_ACTIONS = {"add", "remove", "replace", "rerank", "explain"}
    ALLOWED_ZONES = {"Dream", "Target", "Safe"}

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
        if not str(feedback or "").strip():
            raise ValueError("Feedback cannot be empty.")

        current = self.preferences.normalise(
            previous_result.get("recommendations", [])
        )
        plan = self._create_agent_plan(previous_result, current, feedback)

        notes: list[str] = []
        responses: list[str] = []
        audit: list[dict[str, Any]] = []
        changed = False

        operations = plan.get("operations", [])
        if not isinstance(operations, list):
            operations = []

        for raw_operation in operations:
            if not isinstance(raw_operation, dict):
                continue

            action = str(raw_operation.get("action", "")).strip().casefold()
            if action not in self.SUPPORTED_ACTIONS:
                continue

            if action == "add":
                result = self._execute_agentic_addition(
                    previous_result=previous_result,
                    current=current,
                    feedback=feedback,
                    operation=raw_operation,
                )
                current = result["recommendations"]
                changed = changed or result["changed"]
                notes.extend(result["notes"])
                responses.append(result["response"])
                audit.append(result["audit"])

            elif action == "remove":
                result = self._execute_removal(current, raw_operation)
                current = result["recommendations"]
                changed = changed or result["changed"]
                notes.extend(result["notes"])
                responses.append(result["response"])
                audit.append(result["audit"])

            elif action == "replace":
                result = self._execute_agentic_replacement(
                    previous_result=previous_result,
                    current=current,
                    feedback=feedback,
                    operation=raw_operation,
                )
                current = result["recommendations"]
                changed = changed or result["changed"]
                notes.extend(result["notes"])
                responses.append(result["response"])
                audit.append(result["audit"])

            elif action == "rerank":
                result = self._execute_rerank(current, raw_operation)
                current = result["recommendations"]
                changed = changed or result["changed"]
                notes.extend(result["notes"])
                responses.append(result["response"])
                audit.append(result["audit"])

            elif action == "explain":
                result = self._execute_explanation(
                    previous_result=previous_result,
                    current=current,
                    feedback=feedback,
                    operation=raw_operation,
                )
                responses.append(result["response"])
                notes.extend(result["notes"])
                audit.append(result["audit"])

        if not operations:
            responses.append(
                "I reviewed the feedback, but could not identify a supported counselling "
                "action. The preference list was left unchanged."
            )
            notes.append("No supported action was identified by the feedback agent.")

        final_recommendations = self._classify_and_sort_recommendations(current)
        final_recommendations = self.preferences.resequence(final_recommendations)
        requested_counts = previous_result.get("requested_zone_counts")
        validation = self.preferences.validate(
            final_recommendations,
            requested_counts if isinstance(requested_counts, dict) else None,
        )

        updated = dict(previous_result)
        updated["recommendations"] = final_recommendations
        updated["feedback"] = feedback
        updated["feedback_plan"] = plan
        updated["feedback_notes"] = notes
        updated["counsellor_responses"] = responses
        updated["feedback_audit"] = audit
        updated["list_changed"] = changed
        updated["validation"] = validation
        updated["zone_counts"] = validation.get(
            "zone_counts",
            dict(Counter(str(item.get("zone")) for item in final_recommendations)),
        )
        updated["generated_college_count"] = len(final_recommendations)
        updated["status"] = "success" if validation.get("is_valid", True) else "partial_success"
        updated["summary"] = " ".join(responses).strip() or "Preference list reviewed."
        return updated

    def _execute_agentic_addition(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        requested_count = self._positive_int(operation.get("count"), default=1)
        filters = self._operation_filters(operation)

        evidence = self.counsellor.collect_feedback_evidence(
            student_profile=dict(previous_result.get("student_profile", {})),
            filters=filters,
            user_request=feedback,
            existing_recommendations=current,
            evidence_limit=max(20, requested_count * 10),
        )

        decision = self._reason_about_addition(
            previous_result=previous_result,
            current=current,
            feedback=feedback,
            operation=operation,
            evidence=evidence,
            requested_count=requested_count,
        )

        selected = self._materialise_agent_selection(
            decision=decision,
            evidence=evidence,
            current=current,
            maximum=requested_count,
        )

        before = len(current)
        updated_current, added = self.preferences.add_exact(
            current,
            selected,
            requested_count,
        )

        planned_decision = str(decision.get("decision") or "").strip().casefold()

        if added:
            current = updated_current
            effective_decision = (
                planned_decision
                if planned_decision in {"approve", "approve_with_warning"}
                else "approve_with_warning"
            )
            explanation = self._successful_addition_explanation(
                feedback=feedback,
                added=added,
                decision=decision,
            )
        else:
            effective_decision = "reject"
            explanation = self._addition_not_applied_explanation(
                feedback=feedback,
                decision=decision,
                evidence=evidence,
                selected=selected,
            )

        return {
            "recommendations": current,
            "changed": bool(added),
            "notes": [
                f"Agent selected {len(selected)} grounded option(s); "
                f"{len(added)} unique option(s) were actually added."
            ],
            "response": explanation,
            "audit": {
                "action": "add",
                "requested_count": requested_count,
                "selected_candidate_ids": decision.get("selected_candidate_ids", []),
                "materialised_candidate_count": len(selected),
                "added_count": len(added),
                "before_count": before,
                "after_count": len(current),
                "filters": filters,
                "agent_decision": planned_decision,
                "effective_decision": effective_decision,
                "portfolio_guidance": decision.get("portfolio_guidance"),
                "evidence_candidate_count": len(evidence.get("candidates", [])),
                "added_candidates": [
                    {
                        "college": item.get("college"),
                        "branch": item.get("branch"),
                        "zone": item.get("zone"),
                        "historical_cutoff": item.get("historical_cutoff"),
                    }
                    for item in added
                ],
            },
        }

    def _execute_agentic_replacement(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        remove_field = operation.get("field")
        remove_value = str(operation.get("value", "")).strip()
        replacement_filters = operation.get("replacement_filters")
        if not isinstance(replacement_filters, dict):
            replacement_filters = self._operation_filters(operation)

        proposed_remaining, removed = self.preferences.remove_matching(
            current,
            field=remove_field,
            value=remove_value,
        )
        requested_count = self._positive_int(
            operation.get("count"),
            default=max(1, len(removed)),
        )

        evidence = self.counsellor.collect_feedback_evidence(
            student_profile=dict(previous_result.get("student_profile", {})),
            filters=replacement_filters,
            user_request=feedback,
            existing_recommendations=proposed_remaining,
            evidence_limit=max(20, requested_count * 10),
        )
        decision = self._reason_about_addition(
            previous_result=previous_result,
            current=proposed_remaining,
            feedback=feedback,
            operation={**operation, "action": "replace"},
            evidence=evidence,
            requested_count=requested_count,
        )
        selected = self._materialise_agent_selection(
            decision=decision,
            evidence=evidence,
            current=proposed_remaining,
            maximum=requested_count,
        )
        proposed_final, added = self.preferences.add_exact(
            proposed_remaining,
            selected,
            requested_count,
        )

        # Transactional safety: commit only when a grounded replacement was
        # actually added. Otherwise preserve the original list.
        committed = bool(removed) and bool(added)
        if committed:
            final = proposed_final
            response = self._successful_replacement_explanation(
                removed=removed,
                added=added,
                decision=decision,
            )
        else:
            final = current
            response = (
                "I could not complete the replacement because no grounded new "
                "preference was actually added. The original list was preserved."
            )

        return {
            "recommendations": final,
            "changed": committed,
            "notes": [
                f"Replacement {'committed' if committed else 'rolled back'}: "
                f"{len(removed)} candidate(s) matched removal and {len(added)} grounded replacement(s) were available."
            ],
            "response": response,
            "audit": {
                "action": "replace",
                "committed": committed,
                "removed_count": len(removed) if committed else 0,
                "added_count": len(added) if committed else 0,
                "requested_count": requested_count,
                "replacement_filters": replacement_filters,
                "agent_decision": decision.get("decision"),
            },
        }

    def _execute_removal(
        self,
        current: list[dict[str, Any]],
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        field = operation.get("field")
        value = str(operation.get("value", "")).strip()
        before = len(current)
        updated, removed = self.preferences.remove_matching(
            current,
            field=field,
            value=value,
        )
        remaining = self.preferences.count_matching(updated, field=field, value=value)
        if remaining:
            raise RuntimeError(
                f"Removal validation failed: {remaining} matching recommendation(s) remain."
            )

        removed_names = [
            f"{item.get('college')} - {item.get('branch')} ({item.get('zone')})"
            for item in removed[:10]
        ]
        response = str(operation.get("counsellor_reasoning") or "").strip()
        if removed:
            details = "; ".join(removed_names)
            response = response or (
                f"I removed {len(removed)} matching preference(s): {details}. "
                "Please review the remaining Dream, Target and Safe balance before final submission."
            )
        else:
            response = response or "No existing recommendation matched the removal request, so the list was unchanged."

        return {
            "recommendations": updated,
            "changed": bool(removed),
            "notes": [f"Removed {len(removed)} matching recommendation(s)."],
            "response": response,
            "audit": {
                "action": "remove",
                "field": field,
                "value": value,
                "removed_count": len(removed),
                "before_count": before,
                "after_count": len(updated),
                "remaining_matches": remaining,
            },
        }

    def _execute_rerank(
        self,
        current: list[dict[str, Any]],
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        field = operation.get("field")
        value = str(operation.get("value", "")).strip()
        direction = str(operation.get("direction", "top")).strip().casefold()
        if direction not in {"top", "bottom"}:
            direction = "top"
        updated = self.preferences.rerank(
            current,
            field=field,
            value=value,
            direction=direction,
        )
        response = str(operation.get("counsellor_reasoning") or "").strip() or (
            f"I moved recommendations matching {value or field} toward the {direction}. "
            "Only the ordering changed; eligibility and cutoff evidence were preserved."
        )
        return {
            "recommendations": updated,
            "changed": updated != current,
            "notes": [f"Re-ranked matching recommendations toward the {direction}."],
            "response": response,
            "audit": {
                "action": "rerank",
                "field": field,
                "value": value,
                "direction": direction,
            },
        }

    def _execute_explanation(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        operation: dict[str, Any],
    ) -> dict[str, Any]:
        filters = self._operation_filters(operation)
        evidence = self.counsellor.collect_feedback_evidence(
            student_profile=dict(previous_result.get("student_profile", {})),
            filters=filters,
            user_request=feedback,
            existing_recommendations=current,
            evidence_limit=20,
        )
        response = self.llm.generate_json(
            system_prompt="""
You are an experienced MHT-CET CAP counsellor answering a student's question.
Use only the supplied current portfolio and retrieved CAP evidence.
Explain the cutoff comparison, category/seat eligibility, whether the option is
already present, and the practical risk. Never promise admission. If the option
is outside configured windows, explain that this is a risk signal, not proof that
the student cannot include it in the preference form. Return JSON:
{"answer": "detailed counsellor response"}
""",
            user_prompt=(
                f"QUESTION:\n{feedback}\n\n"
                f"STUDENT PROFILE:\n{json.dumps(previous_result.get('student_profile', {}), default=str)}\n\n"
                f"CURRENT PORTFOLIO:\n{json.dumps(current, default=str)}\n\n"
                f"GROUNDED EVIDENCE:\n{json.dumps(evidence, default=str)}"
            ),
            temperature=0.1,
            max_tokens=1800,
        )
        answer = str(response.get("answer") or "").strip()
        if not answer:
            answer = "I reviewed the indexed evidence but could not produce a reliable explanation."
        return {
            "response": answer,
            "notes": ["Answered without changing the preference list."],
            "audit": {
                "action": "explain",
                "filters": filters,
                "evidence_candidate_count": len(evidence.get("candidates", [])),
            },
        }

    def _create_agent_plan(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
    ) -> dict[str, Any]:
        return self.llm.generate_json(
            system_prompt="""
You are the planning brain of an MHT-CET CAP feedback counsellor.
Interpret the student's natural-language request and choose the counselling tools
needed. You do not invent or edit recommendation rows yourself.

Available actions:
- add: retrieve and evaluate new grounded choices
- remove: remove current choices matching a field/value
- replace: remove matching current choices and retrieve grounded alternatives
- rerank: change ordering only
- explain: answer why an option is or is not recommended without changing the list

Important counselling policy:
1. An explicitly requested college may be considered even when its historical
   cutoff is above the student's percentile or outside configured windows.
2. Do not automatically approve or reject it from cutoff difference alone.
3. Ground every addition in indexed CAP evidence and an eligible seat record.
4. Preserve realistic Target/Safe coverage when giving portfolio guidance.
5. State risk honestly and never promise admission.
6. For add/replace, extract precise filters such as college, branch, location,
   seat_type, or zone. Do not invent omitted filters.
7. For replace, use field/value for what to remove and replacement_filters for
   the requested alternatives.
8. Keep multiple operations in the student's requested order.

Return exactly:
{
  "intent": "modify_list" | "question_only" | "mixed",
  "operations": [
    {
      "action": "add|remove|replace|rerank|explain",
      "count": 1,
      "field": "college|branch|location|zone|seat_type",
      "value": "...",
      "direction": "top|bottom",
      "filters": {},
      "replacement_filters": {},
      "counsellor_reasoning": "brief reason for this operation"
    }
  ],
  "rationale": "planning rationale"
}
""",
            user_prompt=(
                f"STUDENT PROFILE:\n{json.dumps(previous_result.get('student_profile', {}), default=str)}\n\n"
                f"CURRENT LIST COUNT: {len(current)}\n"
                f"CURRENT LIST:\n{json.dumps(current, default=str)}\n\n"
                f"STUDENT FEEDBACK:\n{feedback}"
            ),
            temperature=0.0,
            max_tokens=2000,
        )

    def _reason_about_addition(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        operation: dict[str, Any],
        evidence: dict[str, Any],
        requested_count: int,
    ) -> dict[str, Any]:
        return self.llm.generate_json(
            system_prompt="""
You are the decision-making counsellor for an MHT-CET CAP preference list.
Choose only from the supplied candidate IDs. You may not invent a college,
branch, seat type, cutoff, source, or candidate ID.

Your task is to decide whether and how the requested grounded options should be
added. An option outside the configured percentile window is not automatically
rejected. When the student explicitly requests a high-cutoff college, you may
select it as an aspirational preference if grounded evidence exists and the seat
is eligible. Explain the cutoff difference, admission risk, and why it should not
replace realistic alternatives. Never promise admission.

Evaluate:
- explicit student intent
- historical cutoff versus student percentile
- seat/category eligibility already established in evidence
- duplicate status
- branch/location preferences
- current Dream/Target/Safe balance
- requested number of additions

Return exactly:
{
  "decision": "approve" | "approve_with_warning" | "reject" | "suggest_alternative",
  "selected_candidate_ids": ["candidate-1"],
  "candidate_annotations": [
    {
      "candidate_id": "candidate-1",
      "assigned_zone": "Dream|Target|Safe",
      "risk_level": "Low|Moderate|High|Very High",
      "reasoning": "evidence-based reason shown in the final list"
    }
  ],
  "portfolio_guidance": "how to keep the overall list balanced",
  "counsellor_response": "detailed response to the student"
}

Rules:
- Select no more than the requested count.
- Never select candidates marked already_present=true.
- If no grounded candidate exists, select none and explain why.
- A configured_zone is advisory evidence, not a forced decision.
- Every selected candidate ID must have exactly one candidate_annotations entry.
- Every selected annotation should contain a proposed assigned_zone:
  Dream, Target, or Safe, together with risk reasoning.
- The proposed zone is advisory. Python will calculate the final zone from the
  historical cutoff, student percentile, and configured cutoff windows.
- Never infer a decision from a college name and never use a college-specific rule.
- Do not claim that a row was added or replaced. Python reports whether the
  deterministic mutation actually succeeded.
""",
            user_prompt=(
                f"STUDENT FEEDBACK:\n{feedback}\n\n"
                f"REQUESTED COUNT: {requested_count}\n"
                f"PLANNED OPERATION:\n{json.dumps(operation, default=str)}\n\n"
                f"STUDENT PROFILE:\n{json.dumps(previous_result.get('student_profile', {}), default=str)}\n\n"
                f"CURRENT PORTFOLIO:\n{json.dumps(current, default=str)}\n\n"
                f"NEUTRAL GROUNDED EVIDENCE:\n{json.dumps(evidence, default=str)}"
            ),
            temperature=0.1,
            max_tokens=2600,
        )

    def _materialise_agent_selection(
        self,
        decision: dict[str, Any],
        evidence: dict[str, Any],
        current: list[dict[str, Any]],
        maximum: int,
    ) -> list[dict[str, Any]]:
        decision_name = str(decision.get("decision") or "").strip().casefold()
        if decision_name not in {"approve", "approve_with_warning"}:
            return []

        candidates = evidence.get("candidates", [])
        if not isinstance(candidates, list):
            return []
        by_id = {
            str(item.get("candidate_id")): item
            for item in candidates
            if isinstance(item, dict) and item.get("candidate_id")
        }
        annotations = decision.get("candidate_annotations", [])
        annotation_by_id = {
            str(item.get("candidate_id")): item
            for item in annotations
            if isinstance(item, dict) and item.get("candidate_id")
        }
        selected_ids = decision.get("selected_candidate_ids", [])
        if not isinstance(selected_ids, list):
            return []

        existing_keys = {self.preferences.key(item) for item in current}
        output: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for raw_id in selected_ids:
            candidate_id = str(raw_id)
            if candidate_id in seen_ids or candidate_id not in by_id:
                continue
            source = by_id[candidate_id]
            if source.get("already_present"):
                continue
            if self.preferences.key(source) in existing_keys:
                continue

            annotation = annotation_by_id.get(candidate_id, {})
            assigned_zone, assignment_source = self._classify_candidate_zone(source)

            # The reasoning agent chooses the grounded candidate. Zone placement is
            # then calculated deterministically from the candidate cutoff, student
            # percentile, and configured Dream/Target/Safe windows.
            if assigned_zone not in self.ALLOWED_ZONES:
                continue

            record = dict(source)
            record["zone"] = assigned_zone
            record["zone_assignment_source"] = assignment_source
            record["agent_proposed_zone"] = self._normalise_zone_name(
                annotation.get("assigned_zone")
            )
            record["risk_level"] = str(annotation.get("risk_level") or "Not stated")
            record["reason"] = str(annotation.get("reasoning") or source.get("reason") or "").strip()
            record["agent_selected"] = True
            record["agent_decision"] = str(decision.get("decision") or "")
            record["within_recommendation_window"] = bool(source.get("within_configured_window"))
            record["decision_code"] = "AGENT_SELECTED_FROM_GROUNDED_EVIDENCE"
            output.append(record)
            seen_ids.add(candidate_id)
            if len(output) >= max(1, maximum):
                break

        return output

    def _classify_candidate_zone(
        self,
        candidate: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Classify using configured cutoff windows without college-specific rules.

        First, use CounsellorAgent's configured classification. If an explicitly
        requested option lies outside every configured interval, assign it to the
        nearest configured zone interval. This naturally places very high-cutoff
        options in Dream and substantially lower-cutoff options in Safe.
        """

        try:
            cutoff = float(candidate.get("historical_cutoff"))
            percentile = float(candidate.get("student_percentile"))
        except (TypeError, ValueError):
            return None, "invalid_cutoff_data"

        configured_zone = self.counsellor._classify_zone(cutoff, percentile)
        if configured_zone in self.ALLOWED_ZONES:
            return configured_zone, "configured_cutoff_window"

        margin = cutoff - percentile
        settings = self.counsellor.settings
        intervals = {
            "Dream": (
                float(settings.dream_margin_min),
                float(settings.dream_margin_max),
            ),
            "Target": (
                float(settings.target_margin_min),
                float(settings.target_margin_max),
            ),
            "Safe": (
                float(settings.safe_margin_min),
                float(settings.safe_margin_max),
            ),
        }

        def distance_to_interval(zone: str) -> tuple[float, int]:
            lower, upper = intervals[zone]
            if lower > upper:
                lower, upper = upper, lower
            if margin < lower:
                distance = lower - margin
            elif margin > upper:
                distance = margin - upper
            else:
                distance = 0.0
            # Stable tie-break follows portfolio risk order.
            order = {"Dream": 0, "Target": 1, "Safe": 2}
            return distance, order[zone]

        nearest_zone = min(self.ALLOWED_ZONES, key=distance_to_interval)
        return nearest_zone, "nearest_configured_cutoff_window"

    def _classify_and_sort_recommendations(
        self,
        recommendations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reclassify by cutoff and sort Dream→Target→Safe, cutoff high→low."""

        output: list[dict[str, Any]] = []
        for item in self.preferences.normalise(recommendations):
            record = dict(item)
            zone, source = self._classify_candidate_zone(record)
            if zone in self.ALLOWED_ZONES:
                record["zone"] = zone
                record["zone_assignment_source"] = source
            output.append(record)

        zone_order = {"Dream": 0, "Target": 1, "Safe": 2}

        def cutoff_value(item: dict[str, Any]) -> float:
            try:
                return float(item.get("historical_cutoff"))
            except (TypeError, ValueError):
                return float("-inf")

        output.sort(
            key=lambda item: (
                zone_order.get(str(item.get("zone")), 99),
                -cutoff_value(item),
                str(item.get("college") or "").casefold(),
                str(item.get("branch") or "").casefold(),
            )
        )
        return output

    @classmethod
    def _normalise_zone_name(cls, value: Any) -> str | None:
        normalised = {
            "dream": "Dream",
            "aspirational": "Dream",
            "target": "Target",
            "match": "Target",
            "competitive": "Target",
            "safe": "Safe",
            "safer": "Safe",
            "safety": "Safe",
        }.get(str(value or "").strip().casefold())
        return normalised if normalised in cls.ALLOWED_ZONES else None

    @staticmethod
    def _successful_addition_explanation(
        *,
        feedback: str,
        added: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> str:
        details = "; ".join(
            f"{item.get('college')} - {item.get('branch')} "
            f"({item.get('zone')}, cutoff "
            f"{float(item.get('historical_cutoff', 0.0)):.2f})"
            for item in added
        )
        guidance = str(decision.get("portfolio_guidance") or "").strip()
        warning = str(decision.get("counsellor_response") or "").strip()

        parts = [
            f"I added {len(added)} grounded preference(s) for '{feedback}': {details}."
        ]
        if warning:
            parts.append(warning)
        if guidance:
            parts.append(guidance)
        return " ".join(parts)

    @staticmethod
    def _addition_not_applied_explanation(
        *,
        feedback: str,
        decision: dict[str, Any],
        evidence: dict[str, Any],
        selected: list[dict[str, Any]],
    ) -> str:
        candidates = evidence.get("candidates", [])
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        selected_ids = decision.get("selected_candidate_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []

        if candidate_count == 0:
            return (
                f"I reviewed '{feedback}', but no eligible grounded CAP candidate "
                "was available from the indexed evidence. The list was not changed."
            )

        if selected_ids and not selected:
            return (
                f"I reviewed '{feedback}' and found grounded evidence, but the proposed "
                "candidate could not pass deterministic validation. It may have been "
                "already present, used an invalid candidate ID, or lacked a valid "
                "evidence-based Dream/Target/Safe assignment. No row was added."
            )

        agent_response = str(decision.get("counsellor_response") or "").strip()
        if agent_response:
            return (
                f"{agent_response} However, no new unique recommendation was actually "
                "added, so the list and workbook remain unchanged."
            )

        return (
            f"I reviewed '{feedback}', but no new unique grounded option passed final "
            "validation. The list remains unchanged."
        )

    @staticmethod
    def _successful_replacement_explanation(
        *,
        removed: list[dict[str, Any]],
        added: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> str:
        added_details = "; ".join(
            f"{item.get('college')} - {item.get('branch')} ({item.get('zone')})"
            for item in added
        )
        guidance = str(decision.get("portfolio_guidance") or "").strip()
        response = (
            f"I replaced {len(removed)} existing preference(s) with "
            f"{len(added)} grounded preference(s): {added_details}."
        )
        return f"{response} {guidance}".strip()

    @staticmethod
    def _operation_filters(operation: dict[str, Any]) -> dict[str, Any]:
        filters = operation.get("filters")
        output = dict(filters) if isinstance(filters, dict) else {}
        field = str(operation.get("field") or "").strip().casefold()
        value = str(operation.get("value") or "").strip()
        if field in {"college", "branch", "location", "zone", "seat_type"} and value:
            output.setdefault(field, value)
        return {
            key: value
            for key, value in output.items()
            if value not in (None, "", [], {})
        }

    @staticmethod
    def _positive_int(value: Any, default: int = 1) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, 100))

    @staticmethod
    def _fallback_addition_explanation(
        feedback: str,
        requested_count: int,
        added: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> str:
        if added:
            details = "; ".join(
                f"{item.get('college')} - {item.get('branch')} "
                f"(cutoff {float(item.get('historical_cutoff', 0.0)):.2f}, "
                f"student {float(item.get('student_percentile', 0.0)):.2f}, "
                f"{item.get('risk_level') or item.get('zone')} risk)"
                for item in added
            )
            return (
                f"I added {len(added)} grounded preference(s) for your request '{feedback}': {details}. "
                "Please retain realistic Target and Safe choices because historical cutoffs do not guarantee admission."
            )
        return (
            f"I could not add the requested {requested_count} option(s). "
            f"The indexed evidence produced {len(evidence.get('candidates', []))} eligible candidate record(s), "
            "but none was safely selected as a new unique preference."
        )


_feedback_agent: FeedbackAgent | None = None


def get_feedback_agent() -> FeedbackAgent:
    global _feedback_agent
    if _feedback_agent is None:
        _feedback_agent = FeedbackAgent()
    return _feedback_agent