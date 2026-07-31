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

        pending = previous_result.get("pending_feedback_action")
        if isinstance(pending, dict) and pending.get("status") in {
            "awaiting_confirmation",
            "awaiting_choice",
        }:
            return self._handle_pending_response(
                previous_result=previous_result,
                current=current,
                feedback=feedback,
                pending=pending,
            )

        plan = self._create_agent_plan(previous_result, current, feedback)

        notes: list[str] = []
        responses: list[str] = []
        audit: list[dict[str, Any]] = []
        changed = False
        pending_action: dict[str, Any] | None = None

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
                pending_action = result.get("pending_action")

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

            # A pending counselling decision must be resolved before any later
            # mutation is attempted. This keeps the workflow transactional.
            if pending_action:
                break

        if not operations:
            responses.append(
                "I reviewed the feedback, but could not identify a supported counselling "
                "action. The preference list was left unchanged."
            )
            notes.append("No supported action was identified by the feedback agent.")

        return self._build_result(
            previous_result=previous_result,
            current=current,
            feedback=feedback,
            plan=plan,
            notes=notes,
            responses=responses,
            audit=audit,
            changed=changed,
            pending_action=pending_action,
        )

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
            evidence_limit=max(30, requested_count * 12),
        )
        alternatives = self._collect_alternative_evidence(
            previous_result=previous_result,
            current=current,
            feedback=feedback,
            filters=filters,
            primary_evidence=evidence,
        )

        decision = self._reason_about_addition(
            previous_result=previous_result,
            current=current,
            feedback=feedback,
            operation=operation,
            evidence=evidence,
            alternatives=alternatives,
            requested_count=requested_count,
        )

        selected = self._materialise_agent_selection(
            decision=decision,
            evidence=evidence,
            current=current,
            maximum=requested_count,
            allow_pending=True,
        )
        planned_decision = str(decision.get("decision") or "").strip().casefold()
        requires_confirmation = bool(decision.get("requires_confirmation"))

        # Risky or discouraged requests are never silently rejected and never
        # immediately written to Excel. The grounded option and alternatives are
        # stored so the student's next reply can resolve the decision.
        if selected and (
            requires_confirmation
            or planned_decision in {"approve_with_warning", "suggest_alternative"}
        ):
            pending_action = self._create_pending_action(
                feedback=feedback,
                operation=operation,
                decision=decision,
                selected=selected,
                alternatives=alternatives,
                requested_count=requested_count,
            )
            response = self._pending_counselling_response(
                decision=decision,
                requested=selected,
                alternatives=alternatives,
            )
            return {
                "recommendations": current,
                "changed": False,
                "pending_action": pending_action,
                "notes": [
                    "The requested option was grounded and analysed, but no row was added because student confirmation is required."
                ],
                "response": response,
                "audit": {
                    "action": "add",
                    "status": "awaiting_confirmation",
                    "requested_count": requested_count,
                    "selected_candidate_ids": decision.get("selected_candidate_ids", []),
                    "same_college_alternative_count": len(alternatives.get("same_college", [])),
                    "same_branch_alternative_count": len(alternatives.get("same_branch", [])),
                    "agent_decision": planned_decision,
                    "requires_confirmation": True,
                },
            }

        before = len(current)
        updated_current, added = self.preferences.add_exact(
            current,
            selected,
            requested_count,
        )

        if added:
            current = updated_current
            effective_decision = (
                planned_decision
                if planned_decision in {"approve", "approve_with_warning"}
                else "approve"
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
            "pending_action": None,
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

    def _collect_alternative_evidence(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        filters: dict[str, Any],
        primary_evidence: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        candidates = primary_evidence.get("candidates", [])
        candidates = [item for item in candidates if isinstance(item, dict)]
        first = candidates[0] if candidates else {}

        requested_college = str(filters.get("college") or first.get("college") or "").strip()
        requested_branch = str(filters.get("branch") or first.get("branch") or "").strip()
        output: dict[str, list[dict[str, Any]]] = {
            "same_college": [],
            "same_branch": [],
        }

        if requested_college:
            same_college_evidence = self.counsellor.collect_feedback_evidence(
                student_profile=dict(previous_result.get("student_profile", {})),
                filters={"college": requested_college},
                user_request=(
                    f"Find grounded alternative branches in {requested_college} for: {feedback}"
                ),
                existing_recommendations=current,
                evidence_limit=50,
            )
            rows = same_college_evidence.get("candidates", [])
            output["same_college"] = self._prepare_alternative_candidates(
                rows=rows if isinstance(rows, list) else [],
                prefix="same-college",
                exclude_college="",
                exclude_branch=requested_branch,
                limit=10,
            )

        if requested_branch:
            same_branch_evidence = self.counsellor.collect_feedback_evidence(
                student_profile=dict(previous_result.get("student_profile", {})),
                filters={"branch": requested_branch},
                user_request=(
                    f"Find colleges with {requested_branch} aligned with the student's percentile for: {feedback}"
                ),
                existing_recommendations=current,
                evidence_limit=80,
            )
            rows = same_branch_evidence.get("candidates", [])
            output["same_branch"] = self._prepare_alternative_candidates(
                rows=rows if isinstance(rows, list) else [],
                prefix="same-branch",
                exclude_college=requested_college,
                exclude_branch="",
                limit=12,
            )

        return output

    def _prepare_alternative_candidates(
        self,
        rows: list[Any],
        prefix: str,
        exclude_college: str,
        exclude_branch: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        prepared: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict) or raw.get("already_present"):
                continue
            college = str(raw.get("college") or "").strip()
            branch = str(raw.get("branch") or "").strip()
            if exclude_college and college.casefold() == exclude_college.casefold():
                continue
            if exclude_branch and branch.casefold() == exclude_branch.casefold():
                continue
            key = (
                college.casefold(),
                branch.casefold(),
                str(raw.get("seat_type") or "").casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            item = dict(raw)
            original_id = str(item.get("candidate_id") or len(prepared) + 1)
            item["candidate_id"] = f"{prefix}::{original_id}"
            prepared.append(item)

        def fit_key(item: dict[str, Any]) -> tuple[int, float, float]:
            try:
                cutoff = float(item.get("historical_cutoff"))
                percentile = float(item.get("student_percentile"))
                gap = cutoff - percentile
            except (TypeError, ValueError):
                return (2, float("inf"), float("inf"))
            # Historically attainable options first, then nearest cutoff.
            chance_group = 0 if gap <= 0 else 1
            return (chance_group, abs(gap), -cutoff)

        prepared.sort(key=fit_key)
        return prepared[:limit]

    def _create_pending_action(
        self,
        feedback: str,
        operation: dict[str, Any],
        decision: dict[str, Any],
        selected: list[dict[str, Any]],
        alternatives: dict[str, list[dict[str, Any]]],
        requested_count: int,
    ) -> dict[str, Any]:
        same_college_ids = set(
            str(value) for value in decision.get("same_college_alternative_ids", [])
            if value
        )
        same_branch_ids = set(
            str(value) for value in decision.get("same_branch_alternative_ids", [])
            if value
        )
        same_college = [
            item for item in alternatives.get("same_college", [])
            if not same_college_ids or str(item.get("candidate_id")) in same_college_ids
        ]
        same_branch = [
            item for item in alternatives.get("same_branch", [])
            if not same_branch_ids or str(item.get("candidate_id")) in same_branch_ids
        ]
        return {
            "status": "awaiting_choice" if same_college or same_branch else "awaiting_confirmation",
            "original_feedback": feedback,
            "operation": operation,
            "requested_count": requested_count,
            "requested_candidates": selected,
            "same_college_alternatives": same_college,
            "same_branch_alternatives": same_branch,
            "agent_decision": decision.get("decision"),
            "confirmation_reason": decision.get("confirmation_reason"),
            "portfolio_guidance": decision.get("portfolio_guidance"),
            "counsellor_response": decision.get("counsellor_response"),
        }

    def _pending_counselling_response(
        self,
        decision: dict[str, Any],
        requested: list[dict[str, Any]],
        alternatives: dict[str, list[dict[str, Any]]],
    ) -> str:
        base = str(decision.get("counsellor_response") or "").strip()
        parts = [base] if base else []
        if requested:
            parts.append("Requested option: " + self._format_candidate_list(requested, limit=3))
        same_college = alternatives.get("same_college", [])
        same_branch = alternatives.get("same_branch", [])
        if same_college:
            parts.append(
                "Better-chance branches in the same college: "
                + self._format_candidate_list(same_college, limit=5)
            )
        if same_branch:
            parts.append(
                "Colleges better aligned for the same branch: "
                + self._format_candidate_list(same_branch, limit=5)
            )
        parts.append(
            "Would you like me to add the requested option anyway, choose one of these alternatives, show more suitable options, or leave the list unchanged?"
        )
        return " ".join(part for part in parts if part).strip()

    def _handle_pending_response(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        interpretation = self._parse_ui_action(feedback, pending)
        if interpretation is None:
            interpretation = self.llm.generate_json(
                system_prompt="""
    You are resolving a pending MHT-CET counselling decision. Interpret the student's
    latest reply only against the supplied pending requested option and alternatives.
    Never invent an option.

    Return exactly:
    {
      "action": "confirm_requested" | "choose_same_college" | "choose_same_branch" | "show_more_same_college" | "show_more_same_branch" | "cancel" | "new_request" | "unclear",
      "selected_candidate_ids": ["candidate-id"],
      "answer": "brief response"
    }

    Examples of meaning, not hardcoded wording:
    - agreement to add the original risky option -> confirm_requested
    - choosing another branch in the same institute -> choose_same_college
    - choosing a better college for the same branch -> choose_same_branch
    - asking to see more branches/colleges -> corresponding show_more action
    - declining -> cancel
    - an unrelated new modification -> new_request
    - ambiguous reply -> unclear
    """,
                user_prompt=(
                    f"PENDING DECISION:\n{json.dumps(pending, default=str)}\n\n"
                    f"STUDENT REPLY:\n{feedback}"
                ),
                temperature=0.0,
                max_tokens=1200,
            )
        action = str(interpretation.get("action") or "unclear").strip().casefold()
        selected_ids = interpretation.get("selected_candidate_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []

        pools = {
            "confirm_requested": pending.get("requested_candidates", []),
            "choose_same_college": pending.get("same_college_alternatives", []),
            "choose_same_branch": pending.get("same_branch_alternatives", []),
        }

        if action in pools:
            pool = [item for item in pools[action] if isinstance(item, dict)]
            by_id = {str(item.get("candidate_id")): item for item in pool}
            chosen = [by_id[str(cid)] for cid in selected_ids if str(cid) in by_id]
            if action == "confirm_requested" and not chosen:
                chosen = pool[: self._positive_int(pending.get("requested_count"), 1)]
            if not chosen:
                response = (
                    "I understood that you want to proceed, but I could not match your reply to one of the grounded pending options. "
                    "Please mention the college and branch, or say 'add the requested option'."
                )
                return self._build_result(
                    previous_result=previous_result,
                    current=current,
                    feedback=feedback,
                    plan={"intent": "resolve_pending", "interpretation": interpretation},
                    notes=["Pending choice remained unresolved."],
                    responses=[response],
                    audit=[{"action": "resolve_pending", "status": "unresolved"}],
                    changed=False,
                    pending_action=pending,
                )

            prepared: list[dict[str, Any]] = []
            for source in chosen:
                record = dict(source)
                zone, assignment_source = self._classify_candidate_zone(record)
                if zone not in self.ALLOWED_ZONES:
                    continue
                record["zone"] = zone
                record["zone_assignment_source"] = assignment_source
                record["user_confirmed_override"] = action == "confirm_requested"
                record["decision_code"] = (
                    "USER_CONFIRMED_HIGH_RISK_OPTION"
                    if action == "confirm_requested"
                    else "USER_SELECTED_COUNSELLOR_ALTERNATIVE"
                )
                prepared.append(record)

            updated_current, added = self.preferences.add_exact(
                current,
                prepared,
                max(1, len(prepared)),
            )
            if added:
                label = (
                    "requested high-risk preference"
                    if action == "confirm_requested"
                    else "selected alternative"
                )
                response = (
                    f"I added the {label}: {self._format_candidate_list(added, limit=10)}. "
                    "The list has been reclassified and sorted by zone and cutoff. Historical cutoffs do not guarantee admission."
                )
                guidance = str(pending.get("portfolio_guidance") or "").strip()
                if guidance:
                    response = f"{response} {guidance}"
                return self._build_result(
                    previous_result=previous_result,
                    current=updated_current,
                    feedback=feedback,
                    plan={"intent": "resolve_pending", "interpretation": interpretation},
                    notes=[f"Resolved pending decision and added {len(added)} grounded option(s)."],
                    responses=[response],
                    audit=[{
                        "action": "resolve_pending",
                        "resolution": action,
                        "added_count": len(added),
                    }],
                    changed=True,
                    pending_action=None,
                )

            return self._build_result(
                previous_result=previous_result,
                current=current,
                feedback=feedback,
                plan={"intent": "resolve_pending", "interpretation": interpretation},
                notes=["The selected pending option was already present or failed validation."],
                responses=["The selected grounded option was not added because it is already present or could not pass final validation."],
                audit=[{"action": "resolve_pending", "resolution": action, "added_count": 0}],
                changed=False,
                pending_action=None,
            )

        if action in {"show_more_same_college", "show_more_same_branch"}:
            key = (
                "same_college_alternatives"
                if action == "show_more_same_college"
                else "same_branch_alternatives"
            )
            options = [item for item in pending.get(key, []) if isinstance(item, dict)]
            label = (
                "branches in the same college"
                if action == "show_more_same_college"
                else "colleges for the same branch"
            )
            response = (
                f"Here are the grounded {label}: {self._format_candidate_list(options, limit=12)}. "
                "Tell me which college and branch you want to add, or ask me to add the original option anyway."
            )
            return self._build_result(
                previous_result=previous_result,
                current=current,
                feedback=feedback,
                plan={"intent": "resolve_pending", "interpretation": interpretation},
                notes=["Displayed pending alternatives without changing the list."],
                responses=[response],
                audit=[{"action": "resolve_pending", "resolution": action}],
                changed=False,
                pending_action=pending,
            )

        if action == "cancel":
            return self._build_result(
                previous_result=previous_result,
                current=current,
                feedback=feedback,
                plan={"intent": "resolve_pending", "interpretation": interpretation},
                notes=["Student cancelled the pending addition."],
                responses=["Understood. I did not add the requested option, and the preference list remains unchanged."],
                audit=[{"action": "resolve_pending", "resolution": "cancel"}],
                changed=False,
                pending_action=None,
            )

        if action == "new_request":
            restarted = dict(previous_result)
            restarted.pop("pending_feedback_action", None)
            restarted["awaiting_user_confirmation"] = False
            return self.apply(restarted, feedback)

        answer = str(interpretation.get("answer") or "").strip() or (
            "I still need your choice. Please say whether to add the requested option anyway, choose an alternative branch, choose a better-aligned college for the same branch, or cancel."
        )
        return self._build_result(
            previous_result=previous_result,
            current=current,
            feedback=feedback,
            plan={"intent": "resolve_pending", "interpretation": interpretation},
            notes=["Student reply was ambiguous; pending decision retained."],
            responses=[answer],
            audit=[{"action": "resolve_pending", "resolution": "unclear"}],
            changed=False,
            pending_action=pending,
        )

    @staticmethod
    def _parse_ui_action(
        feedback: str,
        pending: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Resolve trusted UI button commands without another LLM call."""
        text = str(feedback or "").strip()
        prefix = "UI_ACTION::"
        if not text.startswith(prefix):
            return None

        parts = text.split("::", 2)
        command = parts[1].strip().upper() if len(parts) > 1 else ""
        candidate_id = parts[2].strip() if len(parts) > 2 else ""

        mapping = {
            "CONFIRM_REQUESTED": "confirm_requested",
            "CANCEL": "cancel",
            "SHOW_MORE_SAME_COLLEGE": "show_more_same_college",
            "SHOW_MORE_SAME_BRANCH": "show_more_same_branch",
            "CHOOSE_SAME_COLLEGE": "choose_same_college",
            "CHOOSE_SAME_BRANCH": "choose_same_branch",
        }
        action = mapping.get(command)
        if not action:
            return {
                "action": "unclear",
                "selected_candidate_ids": [],
                "answer": "I could not understand that action. Please choose one of the available options.",
            }

        selected = [candidate_id] if candidate_id else []
        return {
            "action": action,
            "selected_candidate_ids": selected,
            "answer": "",
        }

    @staticmethod
    def _candidate_ui_payload(item: dict[str, Any]) -> dict[str, Any]:
        """Return only fields needed by the client-side counselling card."""
        try:
            cutoff = round(float(item.get("historical_cutoff")), 2)
        except (TypeError, ValueError):
            cutoff = None
        try:
            student = round(float(item.get("student_percentile")), 2)
        except (TypeError, ValueError):
            student = None
        gap = round(cutoff - student, 2) if cutoff is not None and student is not None else None
        return {
            "candidate_id": str(item.get("candidate_id") or ""),
            "college": str(item.get("college") or "").strip(),
            "branch": str(item.get("branch") or "").strip(),
            "seat_type": str(item.get("seat_type") or "").strip(),
            "historical_cutoff": cutoff,
            "student_percentile": student,
            "gap": gap,
            "risk_level": str(item.get("risk_level") or "").strip(),
            "zone": str(item.get("zone") or "").strip(),
            "location": str(item.get("location") or "").strip(),
        }

    def _build_feedback_ui(
        self,
        pending_action: dict[str, Any] | None,
        responses: list[str],
    ) -> dict[str, Any] | None:
        if not isinstance(pending_action, dict):
            return None

        requested = [
            self._candidate_ui_payload(item)
            for item in pending_action.get("requested_candidates", [])
            if isinstance(item, dict)
        ]
        same_college = [
            self._candidate_ui_payload(item)
            for item in pending_action.get("same_college_alternatives", [])
            if isinstance(item, dict)
        ]
        same_branch = [
            self._candidate_ui_payload(item)
            for item in pending_action.get("same_branch_alternatives", [])
            if isinstance(item, dict)
        ]

        message = str(pending_action.get("counsellor_response") or "").strip()
        if not message and responses:
            message = str(responses[0]).strip()

        actions: list[dict[str, str]] = []
        if requested:
            actions.append({
                "id": "confirm_requested",
                "label": "➕ Add Anyway",
                "feedback": "UI_ACTION::CONFIRM_REQUESTED",
                "kind": "primary",
            })
        if same_college:
            actions.append({
                "id": "show_same_college",
                "label": "🎓 Better Branches",
                "feedback": "UI_ACTION::SHOW_MORE_SAME_COLLEGE",
                "kind": "secondary",
            })
        if same_branch:
            actions.append({
                "id": "show_same_branch",
                "label": "🏫 Better Colleges",
                "feedback": "UI_ACTION::SHOW_MORE_SAME_BRANCH",
                "kind": "secondary",
            })
        actions.append({
            "id": "cancel",
            "label": "✖ Leave Unchanged",
            "feedback": "UI_ACTION::CANCEL",
            "kind": "secondary",
        })

        return {
            "type": "pending_choice",
            "title": "Admission Analysis",
            "message": message,
            "confirmation_reason": str(pending_action.get("confirmation_reason") or "").strip(),
            "portfolio_guidance": str(pending_action.get("portfolio_guidance") or "").strip(),
            "requested_options": requested,
            "same_college_alternatives": same_college,
            "same_branch_alternatives": same_branch,
            "actions": actions,
        }

    def _build_result(
        self,
        previous_result: dict[str, Any],
        current: list[dict[str, Any]],
        feedback: str,
        plan: dict[str, Any],
        notes: list[str],
        responses: list[str],
        audit: list[dict[str, Any]],
        changed: bool,
        pending_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
        updated["pending_feedback_action"] = pending_action
        updated["feedback_ui"] = self._build_feedback_ui(pending_action, responses)
        updated["awaiting_user_confirmation"] = bool(pending_action)
        updated["status"] = (
            "awaiting_confirmation"
            if pending_action
            else ("success" if validation.get("is_valid", True) else "partial_success")
        )
        updated["summary"] = " ".join(responses).strip() or "Preference list reviewed."
        return updated

    @staticmethod
    def _format_candidate_list(candidates: list[dict[str, Any]], limit: int = 5) -> str:
        formatted: list[str] = []
        for item in candidates[:limit]:
            try:
                cutoff = f"{float(item.get('historical_cutoff')):.2f}"
            except (TypeError, ValueError):
                cutoff = "not available"
            try:
                student = float(item.get("student_percentile"))
                gap = float(item.get("historical_cutoff")) - student
                gap_text = f", gap {gap:+.2f}"
            except (TypeError, ValueError):
                gap_text = ""
            formatted.append(
                f"{item.get('college')} - {item.get('branch')} "
                f"(cutoff {cutoff}{gap_text}, ID {item.get('candidate_id')})"
            )
        return "; ".join(formatted) if formatted else "no grounded alternatives were found"

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
            alternatives={"same_college": [], "same_branch": []},
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
        alternatives: dict[str, list[dict[str, Any]]],
        requested_count: int,
    ) -> dict[str, Any]:
        return self.llm.generate_json(
            system_prompt="""
You are the decision-making counsellor for an MHT-CET CAP preference list.
Choose only from supplied candidate IDs. Never invent a college, branch, seat,
cutoff, source, or candidate ID.

First analyse the exact college/branch requested by the student. If it is poorly
aligned with the student's percentile, do not simply reject it and do not claim
that it was added. Explain the historical cutoff gap and admission risk, then:
- ask whether the student still wants the requested option added as a high-risk preference;
- identify better branches in the same college when grounded alternatives exist;
- identify better colleges for the same branch when grounded alternatives exist;
- offer to suggest options that align more closely with the student's percentile.

The student owns the final preference list. Your professional recommendation may
be negative, but a grounded and eligible requested option can be added after the
student explicitly confirms it.

Set requires_confirmation=true whenever you advise against the requested option,
consider it high/very-high risk, or prefer alternatives. For an ordinary well-
aligned addition, requires_confirmation may be false and decision may be approve.
Do not use college-specific rules or fixed college names.

Return exactly:
{
  "decision": "approve" | "approve_with_warning" | "reject" | "suggest_alternative",
  "requires_confirmation": true,
  "selected_candidate_ids": ["candidate-1"],
  "same_college_alternative_ids": ["same-college::candidate-2"],
  "same_branch_alternative_ids": ["same-branch::candidate-3"],
  "candidate_annotations": [
    {
      "candidate_id": "candidate-1",
      "assigned_zone": "Dream|Target|Safe",
      "risk_level": "Low|Moderate|High|Very High",
      "reasoning": "evidence-based explanation"
    }
  ],
  "confirmation_reason": "why student confirmation is needed",
  "portfolio_guidance": "how to preserve realistic choices",
  "counsellor_response": "clear, detailed response that asks what the student wants to do next"
}

Rules:
- Select no more than requested_count exact requested candidates.
- Never select candidates marked already_present=true.
- If an exact grounded candidate exists but is unrealistic, include its ID in
  selected_candidate_ids and require confirmation rather than silently discarding it.
- Alternative IDs must come only from the supplied alternative arrays.
- Rank alternatives by closeness to the student's percentile and practical chance,
  while respecting the student's category/seat evidence.
- Do not promise admission.
- Python performs the mutation and final zone calculation.
""",
            user_prompt=f"""STUDENT FEEDBACK:
{feedback}

REQUESTED COUNT: {requested_count}
PLANNED OPERATION:
{json.dumps(operation, default=str)}

STUDENT PROFILE:
{json.dumps(previous_result.get('student_profile', {}), default=str)}

CURRENT PORTFOLIO:
{json.dumps(current, default=str)}

EXACT REQUEST EVIDENCE:
{json.dumps(evidence, default=str)}

SAME COLLEGE / DIFFERENT BRANCH OPTIONS:
{json.dumps(alternatives.get('same_college', []), default=str)}

SAME BRANCH / BETTER COLLEGE OPTIONS:
{json.dumps(alternatives.get('same_branch', []), default=str)}""",
            temperature=0.1,
            max_tokens=3200,
        )

    def _materialise_agent_selection(
        self,
        decision: dict[str, Any],
        evidence: dict[str, Any],
        current: list[dict[str, Any]],
        maximum: int,
        allow_pending: bool = False,
    ) -> list[dict[str, Any]]:
        decision_name = str(decision.get("decision") or "").strip().casefold()
        allowed = {"approve", "approve_with_warning"}
        if allow_pending:
            allowed.update({"suggest_alternative", "reject"})
        if decision_name not in allowed:
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