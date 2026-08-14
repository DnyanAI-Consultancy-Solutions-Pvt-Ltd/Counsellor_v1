from __future__ import annotations

import math
from collections import Counter
import re
from typing import Any

from config.settings import get_settings
from rag.retriever import Retriever
from services.llm_service import LLMService, get_llm_service
from services.preference_service import PreferenceService
from services.university_mapping import UniversityMappingService
from tools.eligibility_tool import apply_profile_filters, seat_is_eligible


class CounsellorAgent:
    ZONES = ("Dream", "Target", "Safe")

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or Retriever()
        self.llm = llm_service or get_llm_service()
        self.preferences = PreferenceService()
        self.university_mapping = UniversityMappingService()

    def counsel(
        self,
        student_profile: dict[str, Any],
        user_request: str | None = None,
    ) -> dict[str, Any]:
        student_profile = dict(student_profile or {})
        percentile = self._validate_profile(student_profile)
        plan = self._create_retrieval_plan(student_profile, user_request)

        preferred_university = str(
            student_profile.get("preferred_university") or ""
        ).strip()
        university_resolution = self.university_mapping.resolve(
            preferred_university
        )

        if preferred_university:
            print(f"Requested university       : {preferred_university}")
            print(
                "Matched university         : "
                f"{university_resolution.get('matched_university')}"
            )
            print(
                "Mapped university colleges : "
                f"{len(university_resolution.get('colleges') or [])}"
            )

            if university_resolution.get("status") != "resolved":
                requested_total = self._requested_college_count(student_profile)
                return {
                    "status": "partial_success",
                    "summary": (
                        f"The selected university '{preferred_university}' "
                        "could not be resolved from data/university_college_mapping.xlsx."
                    ),
                    "strategy": "Structured university mapping filter.",
                    "important_notes": [
                        "Check the university mapping file and selected university."
                    ],
                    "student_profile": student_profile,
                    "retrieval_plan": plan,
                    "university_resolution": university_resolution,
                    "recommendations": [],
                    "evidence_count": 0,
                    "zone_counts": {},
                    "requested_zone_counts": self._allocate_zone_counts(requested_total),
                    "missing_zone_counts": self._allocate_zone_counts(requested_total),
                    "requested_college_count": requested_total,
                    "generated_college_count": 0,
                    "complete_requested_count": False,
                }

        base_queries = self._normalise_queries(plan.get("search_queries", []))
        if not base_queries:
            base_queries = self._fallback_queries(student_profile, user_request)

        requested_total = self._requested_college_count(student_profile)
        requested = self._allocate_zone_counts(requested_total)

        raw_results = self.retriever.retrieve_cutoff_pool(
            queries=base_queries,
            top_k_per_query=max(self.settings.results_per_query, 30),
            final_limit=max(self.settings.retrieval_limit, requested_total * 4),
            preferred_university=(
                university_resolution.get("matched_university")
                if preferred_university
                else None
            ),
        )

        university_filtered = raw_results
        university_filter_report = {
            "filter_stage": "chromadb_metadata",
            "requested_university": preferred_university or None,
            "matched_university": university_resolution.get("matched_university"),
            "rows_returned_by_chroma": len(raw_results),
            "note": (
                "University was applied directly as ChromaDB metadata during "
                "cutoff retrieval; no second workbook post-filter was required."
            ),
        }

        branch_filtered = self._filter_branch_results(
            university_filtered,
            student_profile,
        )
        category_filtered = self.retriever.filter_cutoff_candidates(
            branch_filtered,
            category=str(student_profile.get("category", "OPEN")),
            preferred_branches=[],  # branch filtering already performed with diagnostics
        )
        profile_for_filters = dict(student_profile)
        profile_for_filters.pop("preferred_university", None)

        profile_filtered, filter_report = apply_profile_filters(
            category_filtered,
            profile_for_filters,
        )
        # PASS 1: use the configured Dream / Target / Safe windows.
        candidates = self._build_deterministic_candidates(
            profile_filtered,
            student_profile=student_profile,
            student_percentile=percentile,
        )

        standard_candidate_count = len(candidates)
        expanded_candidate_count = 0

        # PASS 2: if strict windows cannot fill the requested portfolio,
        # add the closest remaining grounded/seat-eligible choices from the
        # SAME university, branch and category pool.
        if len(candidates) < requested_total:
            expanded_pool = self._build_deterministic_candidates(
                profile_filtered,
                student_profile=student_profile,
                student_percentile=percentile,
                include_outside_window=True,
            )

            existing_keys = {
                self.preferences.key(item)
                for item in candidates
            }

            fallback_items: list[dict[str, Any]] = []

            for item in expanded_pool:
                key = self.preferences.key(item)

                if key in existing_keys:
                    continue

                if item.get("zone") is not None:
                    continue

                fallback = dict(item)
                cutoff = float(
                    fallback.get("historical_cutoff", 0.0)
                )

                # Preserve the existing 3-zone UI. A fallback below the
                # student's percentile is Safe; above is Dream.
                fallback["zone"] = (
                    "Dream"
                    if cutoff > percentile
                    else "Safe"
                )
                fallback["within_recommendation_window"] = False
                fallback["decision_code"] = (
                    "ELIGIBLE_OUTSIDE_STANDARD_WINDOW"
                )
                fallback["reason"] = (
                    "Portfolio-completion option: this is a grounded, "
                    "seat-eligible CAP choice from the selected university "
                    "and active branch/category filters, but its historical "
                    "cutoff is outside the standard configured percentile "
                    "window. It was added only because the strict-window "
                    "pool could not fill the requested portfolio size."
                )

                fallback_items.append(fallback)
                existing_keys.add(key)

            fallback_items.sort(
                key=self._candidate_priority
            )

            needed = max(
                0,
                requested_total - len(candidates),
            )

            candidates.extend(
                fallback_items[:needed]
            )

            expanded_candidate_count = min(
                needed,
                len(fallback_items),
            )

        recommendations = self._select_balanced_candidates(
            candidates,
            requested=requested,
            requested_total=requested_total,
        )
        recommendations = self.preferences.resequence(recommendations)

        zone_counts = dict(Counter(item["zone"] for item in recommendations))
        missing = self._missing_counts(recommendations, requested)
        complete = len(recommendations) >= requested_total

        diagnostics = {
            "indexed_cutoff_chunks": len(raw_results),
            "rows_matching_branch": len(branch_filtered),
            "rows_matching_category": len(category_filtered),
            "rows_matching_all_active_filters": len(profile_filtered),
            "active_filter_report": filter_report,
            "standard_window_candidates": standard_candidate_count,
            "expanded_fallback_candidates_added": expanded_candidate_count,
            "eligible_unique_candidates": len(candidates),
            "final_selected": len(recommendations),
            "zone_candidate_counts": dict(Counter(item["zone"] for item in candidates)),
            "requested_university": preferred_university or None,
            "matched_university": university_resolution.get("matched_university"),
            "mapped_university_colleges": len(
                university_resolution.get("colleges") or []
            ),
            "rows_after_university_filter": len(university_filtered),
            "university_filter_report": university_filter_report,
        }
        self._log_diagnostics(student_profile, requested_total, diagnostics)

        strategy = (
            "Retrieved broad CAP cutoff evidence, then applied every active profile filter "
            "and configured percentile-zone rule deterministically before ranking the final list."
        )
        result = {
            "status": "success" if complete else "partial_success",
            "summary": "Generated a grounded MHT-CET CAP preference list from indexed cutoff evidence.",
            "strategy": strategy,
            "important_notes": [],
            "student_profile": student_profile,
            "retrieval_plan": plan,
            "university_resolution": university_resolution,
            "recommendations": recommendations,
            "evidence_count": len(profile_filtered),
            "zone_counts": zone_counts,
            "requested_zone_counts": requested,
            "missing_zone_counts": missing,
            "requested_college_count": requested_total,
            "generated_college_count": len(recommendations),
            "complete_requested_count": complete,
            "pipeline_diagnostics": diagnostics,
        }
        if not complete:
            result["evidence_warning"] = (
                f"Requested {requested_total} colleges, but only {len(recommendations)} unique "
                "college-branch choices matched all active profile filters and configured "
                "cutoff windows in the indexed CAP evidence. "
                f"Pipeline counts: {diagnostics}. No college or cutoff was invented."
            )
        elif any(missing.values()):
            result["important_notes"].append(
                "The requested total was completed by using additional eligible choices from "
                "zones that had more evidence; therefore the Dream/Target/Safe split may differ "
                "slightly from the requested ratio."
            )
        return result

    def _fallback_queries(self, profile: dict[str, Any], user_request: str | None) -> list[str]:
        branches = " ".join(self._preferred_branches(profile)) or "engineering"
        category = str(profile.get("category", "OPEN"))
        return [
            f"MHT CET CAP cutoff {branches} {category}",
            f"Maharashtra engineering colleges {branches} cutoff",
            f"{user_request or ''} CAP admission cutoff evidence",
        ]

    def _filter_branch_results(
        self,
        results: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        branches = self._preferred_branches(profile)
        terms = self.retriever._branch_terms(branches)
        if not terms:
            return results
        output: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") or {}
            branch = str(metadata.get("branch") or metadata.get("course") or "").casefold()
            if any(term in branch for term in terms):
                output.append(result)
        return output

    def _build_deterministic_candidates(
        self,
        results: list[dict[str, Any]],
        student_profile: dict[str, Any],
        student_percentile: float,
        include_outside_window: bool = False,
    ) -> list[dict[str, Any]]:
        best_by_choice: dict[tuple[str, str], dict[str, Any]] = {}

        for evidence_id, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            college = str(metadata.get("college") or metadata.get("institute_name") or "").strip()
            branch = str(metadata.get("branch") or metadata.get("course") or "").strip()
            if not college or not branch:
                continue

            seat_cutoffs = result.get("matching_seat_cutoffs") or {}
            for seat_type, values in seat_cutoffs.items():
                if not seat_is_eligible(str(seat_type), student_profile, result):
                    continue
                try:
                    cutoff = float(values.get("cutoff_percentile"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if not math.isfinite(cutoff) or not 0 <= cutoff <= 100:
                    continue
                zone = self._classify_zone(cutoff, student_percentile)
                if zone is None and not include_outside_window:
                    continue

                decision_code = (
                    "ELIGIBLE_WITHIN_RECOMMENDATION_WINDOW"
                    if zone is not None
                    else "OUTSIDE_RECOMMENDATION_WINDOW"
                )
                reason = (
                    self._reason_for_zone(zone, cutoff, student_percentile)
                    if zone is not None
                    else self._reason_outside_window(cutoff, student_percentile)
                )

                item = {
                    "rank": 0,
                    "zone": zone,
                    "college": college,
                    "branch": branch,
                    "location": self._resolve_location(None, college),
                    "category_or_seat_type": str(seat_type).upper(),
                    "seat_allocation": metadata.get("seat_allocation", "Not available"),
                    "historical_cutoff": round(cutoff, 7),
                    "student_percentile": student_percentile,
                    "cutoff_gap": round(student_percentile - cutoff, 7),
                    "reason": reason,
                    "decision_code": decision_code,
                    "within_recommendation_window": zone is not None,
                    "evidence_ids": [evidence_id],
                    "source_page": metadata.get("page_number", "unknown"),
                    "source_file": metadata.get("source_file", "unknown"),
                }
                key = (college.casefold(), branch.casefold())
                current = best_by_choice.get(key)
                if current is None or self._candidate_priority(item) < self._candidate_priority(current):
                    best_by_choice[key] = item

        candidates = list(best_by_choice.values())
        zone_order = {"Dream": 0, "Target": 1, "Safe": 2}
        candidates.sort(
            key=lambda item: (
                zone_order.get(item.get("zone"), 99),
                self._candidate_priority(item),
            )
        )
        return candidates

    def evaluate_requested_options(
        self,
        student_profile: dict[str, Any],
        filters: dict[str, Any] | None,
        user_request: str | None = None,
        existing_recommendations: list[dict[str, Any]] | None = None,
        requested_count: int = 1,
    ) -> dict[str, Any]:
        """Evaluate a targeted feedback request without changing portfolio generation.

        This method uses the same retriever, profile filters, seat eligibility and
        configured Dream/Target/Safe windows as :meth:`counsel`. Records outside
        those windows are retained only for a grounded rejection explanation.
        """

        profile = dict(student_profile or {})
        filters = dict(filters or {})
        percentile = self._validate_profile(profile)

        college = self._first_filter_value(filters.get("college"))
        branch = self._first_filter_value(filters.get("branch"))
        location = self._first_filter_value(filters.get("location"))
        seat_type = self._first_filter_value(filters.get("seat_type"))

        if college:
            profile["preferred_colleges"] = [college]
        if branch:
            profile["preferred_branches"] = [branch]
        if location:
            profile["preferred_locations"] = [location]
        if seat_type:
            profile["seat_type"] = seat_type

        filter_text = " ".join(
            str(value)
            for value in filters.values()
            if value not in (None, "", [], {})
        ).strip()
        category = str(profile.get("category", "OPEN"))
        queries = self._normalise_queries([
            f"{filter_text} {category} MHT CET CAP cutoff",
            f"{user_request or filter_text} Maharashtra engineering CAP cutoff",
            f"{filter_text} engineering institute branch cutoff percentile",
        ])
        if not queries:
            queries = self._fallback_queries(profile, user_request)

        preferred_university = str(
            profile.get("preferred_university") or ""
        ).strip()
        university_resolution = self.university_mapping.resolve(
            preferred_university
        )

        raw_results = self.retriever.retrieve_cutoff_pool(
            queries=queries,
            top_k_per_query=max(self.settings.results_per_query, 30),
            final_limit=max(self.settings.retrieval_limit, requested_count * 20, 100),
            preferred_university=(
                university_resolution.get("matched_university")
                if preferred_university
                and university_resolution.get("status") == "resolved"
                else None
            ),
        )
        branch_filtered = self._filter_branch_results(raw_results, profile)
        category_filtered = self.retriever.filter_cutoff_candidates(
            branch_filtered,
            category=category,
            preferred_branches=[],
        )
        profile_filtered, filter_report = apply_profile_filters(
            category_filtered,
            profile,
        )

        evaluated = self._build_deterministic_candidates(
            profile_filtered,
            student_profile=profile,
            student_percentile=percentile,
            include_outside_window=True,
        )
        evaluated = [
            item for item in evaluated
            if self.preferences.matches_filters(item, filters)
        ]

        existing_keys = {
            self.preferences.key(item)
            for item in (existing_recommendations or [])
            if isinstance(item, dict)
        }
        qualified = [
            item for item in evaluated
            if item.get("zone") in self.ZONES
            and self.preferences.key(item) not in existing_keys
        ]
        qualified.sort(key=self._candidate_priority)

        outside = [item for item in evaluated if item.get("zone") is None]
        outside.sort(key=self._candidate_priority)
        closest = sorted(evaluated, key=self._candidate_priority)[:5]

        if qualified:
            decision = "approved"
            decision_code = "APPROVED_GROUNDED_OPTION"
        elif not profile_filtered:
            decision = "rejected"
            decision_code = "REQUESTED_COMBINATION_NOT_FOUND"
        elif not evaluated:
            decision = "rejected"
            decision_code = "NO_ELIGIBLE_SEAT"
        elif outside:
            decision = "rejected"
            decision_code = "OUTSIDE_RECOMMENDATION_WINDOW"
        else:
            decision = "rejected"
            decision_code = "ALREADY_PRESENT_OR_FILTER_MISMATCH"

        explanation = self._targeted_decision_explanation(
            decision=decision,
            decision_code=decision_code,
            filters=filters,
            percentile=percentile,
            qualified=qualified,
            closest=closest,
            matching_rows=len(profile_filtered),
        )

        return {
            "decision": decision,
            "decision_code": decision_code,
            "explanation": explanation,
            "qualified_candidates": qualified[: max(1, int(requested_count))],
            "matching_evidence_count": len(profile_filtered),
            "eligible_evidence_count": len(evaluated),
            "closest_evidence": closest,
            "outside_window_evidence": outside[:5],
            "filter_report": filter_report,
            "temporary_profile_filters": {
                key: profile.get(key)
                for key in (
                    "preferred_colleges",
                    "preferred_branches",
                    "preferred_locations",
                    "seat_type",
                )
                if profile.get(key) not in (None, "", [], {})
            },
        }

    def collect_feedback_evidence(
        self,
        student_profile: dict[str, Any],
        filters: dict[str, Any] | None,
        user_request: str | None = None,
        existing_recommendations: list[dict[str, Any]] | None = None,
        evidence_limit: int = 30,
    ) -> dict[str, Any]:
        """Return neutral, grounded evidence for feedback-agent reasoning.

        This method deliberately does not approve, reject, add, remove, classify risk,
        or mutate the recommendation list. It only retrieves indexed CAP records,
        applies profile/seat eligibility, calculates cutoff facts, and returns candidate
        IDs that a reasoning agent may select.
        """

        profile = dict(student_profile or {})
        request_filters = dict(filters or {})
        percentile = self._validate_profile(profile)

        college = self._first_filter_value(request_filters.get("college"))
        branch = self._first_filter_value(request_filters.get("branch"))
        location = self._first_filter_value(request_filters.get("location"))
        seat_type = self._first_filter_value(request_filters.get("seat_type"))

        if college:
            profile["preferred_colleges"] = [college]
        if branch:
            profile["preferred_branches"] = [branch]
        if location:
            profile["preferred_locations"] = [location]
        if seat_type:
            profile["seat_type"] = seat_type

        filter_text = " ".join(
            str(value)
            for value in request_filters.values()
            if value not in (None, "", [], {})
        ).strip()
        category = str(profile.get("category", "OPEN"))
        queries = self._normalise_queries([
            f"{filter_text} {category} MHT CET CAP cutoff",
            f"{user_request or filter_text} Maharashtra engineering CAP cutoff",
            f"{filter_text} engineering institute branch seat cutoff percentile",
        ])
        if not queries:
            queries = self._fallback_queries(profile, user_request)

        preferred_university = str(
            profile.get("preferred_university") or ""
        ).strip()
        university_resolution = self.university_mapping.resolve(
            preferred_university
        )

        raw_results = self.retriever.retrieve_cutoff_pool(
            queries=queries,
            top_k_per_query=max(self.settings.results_per_query, 40),
            final_limit=max(self.settings.retrieval_limit, evidence_limit * 10, 150),
            preferred_university=(
                university_resolution.get("matched_university")
                if preferred_university
                and university_resolution.get("status") == "resolved"
                else None
            ),
        )
        branch_filtered = self._filter_branch_results(raw_results, profile)
        category_filtered = self.retriever.filter_cutoff_candidates(
            branch_filtered,
            category=category,
            preferred_branches=[],
        )
        profile_filtered, filter_report = apply_profile_filters(
            category_filtered,
            profile,
        )

        candidates = self._build_deterministic_candidates(
            profile_filtered,
            student_profile=profile,
            student_percentile=percentile,
            include_outside_window=True,
        )
        candidates = [
            item for item in candidates
            if self.preferences.matches_filters(item, request_filters)
        ]

        existing_keys = {
            self.preferences.key(item)
            for item in (existing_recommendations or [])
            if isinstance(item, dict)
        }

        neutral_candidates: list[dict[str, Any]] = []
        for index, item in enumerate(candidates[: max(1, int(evidence_limit))], start=1):
            record = dict(item)
            record["candidate_id"] = f"candidate-{index}"
            record["already_present"] = self.preferences.key(record) in existing_keys
            cutoff = float(record.get("historical_cutoff", 0.0))
            difference = cutoff - percentile
            record["cutoff_difference"] = round(difference, 7)
            record["cutoff_relation"] = (
                "above_student" if difference > 0
                else "below_student" if difference < 0
                else "equal_to_student"
            )
            # The configured window is factual context, not an approval decision.
            record["configured_zone"] = record.get("zone")
            record["within_configured_window"] = bool(
                record.get("within_recommendation_window")
            )
            neutral_candidates.append(record)

        return {
            "student_percentile": percentile,
            "request_filters": request_filters,
            "search_queries": queries,
            "matching_profile_rows": len(profile_filtered),
            "eligible_candidate_count": len(candidates),
            "candidates": neutral_candidates,
            "filter_report": filter_report,
            "evidence_policy": {
                "source_required": True,
                "eligible_seat_required": True,
                "outside_window_is_not_automatic_rejection": True,
                "agent_must_explain_risk": True,
            },
        }

    @staticmethod
    def _first_filter_value(value: Any) -> str | None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
            return None
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _reason_outside_window(cutoff: float, student_percentile: float) -> str:
        difference = cutoff - student_percentile
        relation = "above" if difference >= 0 else "below"
        return (
            f"Historical cutoff is {abs(difference):.2f} percentile points "
            f"{relation} the student percentile and falls outside the configured "
            "Dream, Target and Safe recommendation windows."
        )

    @staticmethod
    def _targeted_decision_explanation(
        *,
        decision: str,
        decision_code: str,
        filters: dict[str, Any],
        percentile: float,
        qualified: list[dict[str, Any]],
        closest: list[dict[str, Any]],
        matching_rows: int,
    ) -> str:
        requested_text = ", ".join(
            f"{key.replace('_', ' ')} '{value}'"
            for key, value in filters.items()
            if value not in (None, "", [], {})
        ) or "the requested option"

        if decision == "approved" and qualified:
            best = qualified[0]
            return (
                f"The request for {requested_text} is supported by indexed CAP evidence. "
                f"The closest approved option is {best.get('college')} - "
                f"{best.get('branch')} with historical cutoff "
                f"{float(best.get('historical_cutoff', 0.0)):.2f}, compared with "
                f"the student's {percentile:.2f} percentile. It falls in the "
                f"{best.get('zone')} zone, so it can be added."
            )
        if decision_code == "REQUESTED_COMBINATION_NOT_FOUND":
            return (
                f"I did not add {requested_text}. No matching college/branch record "
                "was found after applying the request to the currently indexed CAP "
                "evidence. No college or cutoff was invented."
            )
        if decision_code == "NO_ELIGIBLE_SEAT":
            return (
                f"I found {matching_rows} matching CAP record(s) for {requested_text}, "
                "but none contained a seat allocation eligible for the student's "
                "category, gender and home-university profile. Therefore it was not added."
            )
        if decision_code == "OUTSIDE_RECOMMENDATION_WINDOW" and closest:
            best = closest[0]
            cutoff = float(best.get("historical_cutoff", 0.0))
            gap = cutoff - percentile
            relation = "above" if gap >= 0 else "below"
            return (
                f"I evaluated {requested_text} using indexed CAP evidence, but did not "
                f"add it. The closest eligible record is {best.get('college')} - "
                f"{best.get('branch')} ({best.get('category_or_seat_type')}) with "
                f"historical cutoff {cutoff:.2f}. This is {abs(gap):.2f} percentile "
                f"points {relation} the student's {percentile:.2f} percentile and falls "
                "outside the configured Dream, Target and Safe windows."
            )
        return (
            f"I did not add {requested_text}. Matching evidence was reviewed, but no "
            "new unique option satisfied every requested filter and the configured "
            "counselling criteria. The existing list was left unchanged."
        )

    @staticmethod
    def _candidate_priority(item: dict[str, Any]) -> tuple[float, float]:
        # Prefer choices closest to the student's percentile; then higher cutoffs.
        return (abs(float(item.get("cutoff_gap", 0.0))), -float(item.get("historical_cutoff", 0.0)))

    @staticmethod
    def _reason_for_zone(zone: str, cutoff: float, student_percentile: float) -> str:
        margin = cutoff - student_percentile
        if zone == "Dream":
            return f"Aspirational option: historical cutoff is {margin:.2f} percentile points above the student percentile."
        if zone == "Target":
            return f"Well-aligned option: historical cutoff is within {abs(margin):.2f} percentile points of the student percentile."
        return f"Safer option: historical cutoff is {abs(margin):.2f} percentile points below the student percentile."

    def _select_balanced_candidates(
        self,
        candidates: list[dict[str, Any]],
        requested: dict[str, int],
        requested_total: int,
    ) -> list[dict[str, Any]]:
        by_zone = {zone: [x for x in candidates if x.get("zone") == zone] for zone in self.ZONES}
        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()

        for zone in self.ZONES:
            for item in by_zone[zone][: requested.get(zone, 0)]:
                key = self.preferences.key(item)
                if key not in selected_keys:
                    selected.append(item)
                    selected_keys.add(key)

        # Fill any remaining total from unused eligible choices, rather than returning
        # a tiny list merely because one zone lacks its exact quota.
        if len(selected) < requested_total:
            leftovers = [x for x in candidates if self.preferences.key(x) not in selected_keys]
            leftovers.sort(key=self._candidate_priority)
            for item in leftovers:
                selected.append(item)
                selected_keys.add(self.preferences.key(item))
                if len(selected) >= requested_total:
                    break

        zone_order = {zone: index for index, zone in enumerate(self.ZONES)}
        selected.sort(key=lambda x: (zone_order.get(str(x.get("zone")), 99), self._candidate_priority(x)))
        return selected[:requested_total]

    @staticmethod
    def _log_diagnostics(profile: dict[str, Any], requested_total: int, diagnostics: dict[str, Any]) -> None:
        print("\n" + "=" * 64)
        print("COUNSELLOR PIPELINE DIAGNOSTICS")
        print("=" * 64)
        print(f"Student percentile       : {profile.get('percentile')}")
        print(f"Category                 : {profile.get('category')}")
        print(f"Preferred branches       : {profile.get('preferred_branches') or profile.get('preferred_branch')}")
        print(f"Requested colleges       : {requested_total}")
        print(f"Requested university     : {diagnostics.get('requested_university')}")
        print(f"Matched university       : {diagnostics.get('matched_university')}")
        print(f"Mapped university colleges: {diagnostics.get('mapped_university_colleges')}")
        print(f"Total CAP records        : {diagnostics.get('indexed_cutoff_chunks')}")
        print(f"After university filter  : {diagnostics.get('rows_after_university_filter')}")
        print(f"After branch filter      : {diagnostics.get('rows_matching_branch')}")
        print(f"After category filter    : {diagnostics.get('rows_matching_category')}")
        print(f"After profile filter     : {diagnostics.get('rows_matching_all_active_filters')}")
        print(f"Active filter report     : {diagnostics.get('active_filter_report')}")
        print(f"University filter report : {diagnostics.get('university_filter_report')}")
        print(f"Standard-window candidates: {diagnostics.get('standard_window_candidates')}")
        print(f"Fallback candidates added : {diagnostics.get('expanded_fallback_candidates_added')}")
        print(f"Eligible candidates      : {diagnostics.get('eligible_unique_candidates')}")
        print(f"Zone candidate counts    : {diagnostics.get('zone_candidate_counts')}")
        print(f"Final recommendations    : {diagnostics.get('final_selected')}")
        print("=" * 64 + "\n")

    def _create_retrieval_plan(self, profile: dict[str, Any], user_request: str | None) -> dict[str, Any]:
        return self.llm.generate_json(
            system_prompt="""
You are the retrieval planner for an MHT-CET CAP admissions counsellor.
Create broad semantic searches for cutoff evidence.
Use branch, category or seat type, gender, home university, location, institute type and seat allocation.
IMPORTANT: do not include the student's percentile, percentile windows, exact cutoff numbers, Dream/Target/Safe labels, or arithmetic ranges in any search query. Percentile comparison happens later in a deterministic portfolio tool.
Do not select colleges and do not assume college names unless the student explicitly requested them.
Return {"search_queries": ["..."], "rationale": "..."}.
""",
            user_prompt=(
                f"STUDENT PROFILE:\n{self._format_profile(profile)}\n\n"
                f"ADDITIONAL REQUEST:\n{user_request or 'None'}"
            ),
            temperature=0.1,
            max_tokens=1800,
        )

    def _generate_recommendations(
        self,
        student_profile: dict[str, Any],
        user_request: str | None,
        retrieval_plan: dict[str, Any],
        context: str,
        missing_counts: dict[str, int],
        existing: list[dict[str, Any]],
    ) -> dict[str, Any]:
        existing_keys = [f"{x.get('college')} | {x.get('branch')}" for x in existing]
        return self.llm.generate_json(
            system_prompt=f"""
You are a highly experienced MHT-CET CAP admissions counsellor.
Generate only the still-missing grounded choices:
- Dream: {missing_counts.get('Dream', 0)}
- Target: {missing_counts.get('Target', 0)}
- Safe: {missing_counts.get('Safe', 0)}

Rules:
1. Use only retrieved evidence.
2. Never invent a college, branch, seat type, cutoff, page or evidence ID.
3. Historical cutoff must be copied from a Cutoff Percentile value.
4. Merit Rank must never be used as percentile.
5. Each recommendation must identify the exact selected seat type.
6. One recommendation is one unique college-and-branch choice.
7. The selected category is a strict eligibility filter. Return only a seat type that matches the student's selected category.
   For example, OBC must use an OBC seat code; never return OPEN, SC, ST, EWS, TFWS, DEF or PWD seats for an OBC-only query.
8. Do not return any existing choice listed by the user prompt.
9. Copy the exact cutoff and seat code from evidence. Python will deterministically classify the zone from configuration; your proposed zone is advisory only.
10. Do not apply or invent percentile bands.
11. Do not promise admission.
12. If evidence cannot support all missing choices, return only grounded options.
13. Derive location from the institute name when the evidence has no separate location field.

Return:
{{
  "status": "success",
  "summary": "...",
  "strategy": "...",
  "recommendations": [
    {{
      "rank": 1,
      "zone": "Dream",
      "college": "...",
      "branch": "...",
      "location": "Not available",
      "category_or_seat_type": "...",
      "seat_allocation": "...",
      "historical_cutoff": 95.42,
      "student_percentile": 94.10,
      "cutoff_gap": -1.32,
      "reason": "...",
      "evidence_ids": [1]
    }}
  ],
  "important_notes": ["..."]
}}
""",
            user_prompt=(
                f"STUDENT PROFILE:\n{self._format_profile(student_profile)}\n\n"
                f"ADDITIONAL REQUEST:\n{user_request or 'None'}\n\n"
                f"EXISTING CHOICES TO EXCLUDE:\n{existing_keys}\n\n"
                f"RETRIEVAL PLAN:\n{retrieval_plan}\n\n"
                f"RETRIEVED EVIDENCE:\n{context}"
            ),
            temperature=0.1,
            max_tokens=min(self.settings.llm_max_tokens, 3000),
        )

    def _validate_recommendations(
        self,
        recommendations: Any,
        student_percentile: float,
        student_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(recommendations, list):
            return []
        grouped: dict[str, list[dict[str, Any]]] = {zone: [] for zone in self.ZONES}
        seen: set[tuple[str, str]] = set()
        selected_category = str(student_profile.get("category", "OPEN") or "OPEN").strip().upper()

        for raw in recommendations:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            try:
                cutoff = float(item.get("historical_cutoff"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(cutoff) or not 0 <= cutoff <= 100:
                continue
            college = str(item.get("college", "")).strip()
            branch = str(item.get("branch", "")).strip()
            seat_type = str(item.get("category_or_seat_type", "")).strip().upper()
            zone = self._classify_zone(cutoff, student_percentile)
            evidence_ids = item.get("evidence_ids")
            if not all([college, branch, seat_type, zone]):
                continue
            if not self._seat_matches_category(seat_type, selected_category):
                continue
            if not isinstance(evidence_ids, list) or not evidence_ids:
                continue
            key = (college.casefold(), branch.casefold())
            if key in seen:
                continue
            item["zone"] = zone
            item["category_or_seat_type"] = seat_type
            item["location"] = self._resolve_location(item.get("location"), college)
            item["historical_cutoff"] = round(cutoff, 7)
            item["student_percentile"] = student_percentile
            item["cutoff_gap"] = round(student_percentile - cutoff, 7)
            grouped[zone].append(item)
            seen.add(key)

        return grouped["Dream"] + grouped["Target"] + grouped["Safe"]

    @staticmethod
    def _seat_matches_category(seat_type: str, category: str) -> bool:
        code = re.sub(r"[^A-Z0-9]", "", str(seat_type or "").upper())
        cat = re.sub(r"[^A-Z0-9]", "", str(category or "OPEN").upper())
        if not code:
            return False

        # These are special-purpose allocations, not the student's ordinary category seat.
        if any(token in code for token in ("DEF", "PWD", "TFWS")):
            return False

        category_tokens = {
            "OPEN": ("OPEN",),
            "OBC": ("OBC",),
            "SC": ("SC",),
            "ST": ("ST",),
            "EWS": ("EWS",),
            "SBC": ("SBC",),
            "VJ": ("VJ", "DT"),
            "NTA": ("NTA", "VJ"),
            "NTB": ("NTB",),
            "NTC": ("NTC",),
            "NTD": ("NTD",),
        }
        tokens = category_tokens.get(cat, (cat,))
        return any(token and token in code for token in tokens)

    @staticmethod
    def _resolve_location(location: Any, college: str) -> str:
        supplied = str(location or "").strip()
        if supplied and supplied.casefold() not in {"not available", "n/a", "none", "unknown"}:
            return supplied

        text = re.sub(r"[^a-z0-9]+", " ", str(college or "").casefold())
        # More specific names must be checked before their parent city names.
        locations = (
            ("Navi Mumbai", ("navi mumbai", "airoli", "kamothe", "panvel", "nerul", "vashi", "kharghar")),
            ("Pune", ("pune", "pimpri", "chinchwad", "akurdi", "lavale", "pirangut", "wagholi", "talegaon", "lonavala", "baramati", "haveli", "pisoli")),
            ("Mumbai", ("mumbai", "dombivali", "dombivli", "kalyan", "thane", "bandra", "andheri", "chembur")),
            ("Nagpur", ("nagpur",)),
            ("Nashik", ("nashik", "nasik")),
            ("Kolhapur", ("kolhapur",)),
            ("Sangli", ("sangli", "miraj")),
            ("Satara", ("satara", "karad")),
            ("Chhatrapati Sambhajinagar", ("chhatrapati sambhajinagar", "aurangabad")),
            ("Amravati", ("amravati",)),
            ("Solapur", ("solapur",)),
            ("Ahmednagar", ("ahmednagar", "ahilyanagar")),
        )
        for canonical, aliases in locations:
            if any(alias in text for alias in aliases):
                return canonical
        return "Not available"

    def _merge_recommendations(
        self,
        current: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        limits: dict[str, int],
    ) -> list[dict[str, Any]]:
        output = self.preferences.normalise(current)
        seen = {self.preferences.key(item) for item in output}
        counts = Counter(item.get("zone") for item in output)
        for item in candidates:
            zone = item.get("zone")
            key = self.preferences.key(item)
            if zone not in limits or key in seen or counts[zone] >= limits[zone]:
                continue
            output.append(item)
            seen.add(key)
            counts[zone] += 1
        zone_order = {zone: index for index, zone in enumerate(self.ZONES)}
        output.sort(key=lambda x: (zone_order.get(str(x.get("zone")), 99), -float(x.get("historical_cutoff", 0) or 0)))
        return self.preferences.resequence(output)

    def _completion_queries(
        self,
        base_queries: list[str],
        profile: dict[str, Any],
        user_request: str | None,
        missing: dict[str, int],
        attempt: int,
    ) -> list[str]:
        if attempt == 0:
            return base_queries
        profile_text = " ".join(
            f"{k} {v}" for k, v in profile.items()
            if k not in {"percentile", "college_count"} and v not in (None, "", [], {})
        )
        missing_zones = " ".join(zone for zone, count in missing.items() if count)
        extras = [
            f"{profile_text} {missing_zones} MHT CET CAP cutoff alternatives",
            f"{profile_text} related engineering branches diverse colleges historical cutoff",
            f"{user_request or ''} {missing_zones} unique institute branch choices",
        ]
        return self._normalise_queries(base_queries + extras)

    def _requested_college_count(self, profile: dict[str, Any]) -> int:
        raw = profile.get("college_count", self.settings.recommendation_count)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = self.settings.recommendation_count
        return max(1, min(value, 100))

    def _allocate_zone_counts(self, total: int) -> dict[str, int]:
        """Allocate slots from configurable Dream/Target/Safe ratios."""
        if total <= 1:
            return {"Dream": 0, "Target": total, "Safe": 0}
        ratios = {
            "Dream": max(0.0, float(self.settings.dream_ratio)),
            "Target": max(0.0, float(self.settings.target_ratio)),
            "Safe": max(0.0, float(self.settings.safe_ratio)),
        }
        ratio_sum = sum(ratios.values()) or 1.0
        raw = {zone: total * ratio / ratio_sum for zone, ratio in ratios.items()}
        allocated = {zone: int(value) for zone, value in raw.items()}
        remaining = total - sum(allocated.values())
        for zone in sorted(raw, key=lambda z: raw[z] - allocated[z], reverse=True):
            if remaining <= 0:
                break
            allocated[zone] += 1
            remaining -= 1
        return allocated

    def _classify_zone(self, cutoff: float, student_percentile: float) -> str | None:
        """Classify numerically using configuration, not LLM judgement or retrieval text."""
        margin = cutoff - student_percentile
        windows = (
            ("Dream", self.settings.dream_margin_min, self.settings.dream_margin_max),
            ("Target", self.settings.target_margin_min, self.settings.target_margin_max),
            ("Safe", self.settings.safe_margin_min, self.settings.safe_margin_max),
        )
        for zone, lower, upper in windows:
            # Lower inclusive; upper inclusive only for the last matching boundary.
            if float(lower) <= margin <= float(upper):
                return zone
        return None

    @staticmethod
    def _preferred_branches(profile: dict[str, Any]) -> list[str]:
        branches = profile.get("preferred_branches")
        if isinstance(branches, list) and branches:
            return [str(value) for value in branches if str(value).strip()]
        single = str(profile.get("preferred_branch") or "").strip()
        return [single] if single else []

    @staticmethod
    def _missing_counts(recommendations: list[dict[str, Any]], requested: dict[str, int]) -> dict[str, int]:
        counts = Counter(str(item.get("zone", "")) for item in recommendations)
        return {zone: max(0, required - counts.get(zone, 0)) for zone, required in requested.items()}

    def _normalise_queries(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            text = " ".join(value.split())
            key = text.casefold()
            if text and key not in seen:
                output.append(text)
                seen.add(key)
            if len(output) >= self.settings.max_search_queries:
                break
        return output

    @staticmethod
    def _validate_profile(profile: dict[str, Any]) -> float:
        try:
            percentile = float(profile["percentile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A numeric percentile is required.") from exc
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")
        return percentile

    @staticmethod
    def _normalise_zone(value: Any) -> str | None:
        return {
            "dream": "Dream", "aspirational": "Dream",
            "target": "Target", "match": "Target",
            "safe": "Safe", "safety": "Safe",
        }.get(str(value or "").strip().casefold())

    @staticmethod
    def _format_retrieval_profile(profile: dict[str, Any]) -> str:
        return "\n".join(
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in profile.items()
            if key not in {"percentile", "college_count"} and value not in (None, "", [], {})
        )

    @staticmethod
    def _format_profile(profile: dict[str, Any]) -> str:
        return "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in profile.items())


_agent: CounsellorAgent | None = None


def get_counsellor_agent() -> CounsellorAgent:
    global _agent
    if _agent is None:
        _agent = CounsellorAgent()
    return _agent