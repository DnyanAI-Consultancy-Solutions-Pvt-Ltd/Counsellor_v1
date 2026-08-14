from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from config.settings import get_settings
from rag.store import ChromaStore


class Retriever:
    def __init__(self, store: ChromaStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or ChromaStore()

    def retrieve_multiple(
        self,
        queries: list[str],
        top_k_per_query: int | None = None,
        final_limit: int | None = None,
        document_type: str | None = None,
        university: str | None = None,
    ) -> list[dict[str, Any]]:
        unique_queries = self._clean_queries(queries)
        combined: dict[str, dict[str, Any]] = {}

        result_limit = final_limit or self.settings.retrieval_limit

        for query in unique_queries:
            results = self.store.search(
                query=query,
                top_k=top_k_per_query or self.settings.results_per_query,
                document_type=document_type,
                university=university,
            )

            for result in results:
                key = self._result_key(result)
                existing = combined.get(key)

                existing_queries = (
                    existing.get("matched_queries", [])
                    if existing
                    else []
                )

                candidate = {
                    **result,
                    "matched_queries": sorted(
                        set(existing_queries + [query])
                    ),
                }

                candidate_similarity = float(
                    candidate.get("similarity", 0.0) or 0.0
                )
                existing_similarity = float(
                    existing.get("similarity", 0.0) or 0.0
                ) if existing else 0.0

                if existing is None or candidate_similarity > existing_similarity:
                    combined[key] = candidate
                else:
                    existing["matched_queries"] = candidate["matched_queries"]

        ranked = sorted(
            combined.values(),
            key=lambda item: (
                len(item.get("matched_queries", [])),
                float(item.get("similarity", 0.0) or 0.0),
            ),
            reverse=True,
        )

        return self._diversify(
            results=ranked,
            final_limit=result_limit,
        )

    def _diversify(
        self,
        results: list[dict[str, Any]],
        final_limit: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        college_counts: defaultdict[str, int] = defaultdict(int)

        max_per_college = max(3, final_limit // 12)

        for result in results:
            metadata = result.get("metadata") or {}

            college = str(
                metadata.get("college")
                or metadata.get("institute_name")
                or ""
            ).strip().casefold()

            if college and college_counts[college] >= max_per_college:
                continue

            selected.append(result)

            if college:
                college_counts[college] += 1

            if len(selected) >= final_limit:
                break

        return selected



    def retrieve_cutoff_pool(
        self,
        queries: list[str],
        top_k_per_query: int | None = None,
        final_limit: int | None = None,
        preferred_university: str | None = None,
    ) -> list[dict[str, Any]]:
        """Combine semantic results with the complete structured cutoff pool.

        Semantic hits are placed first, while exhaustive structured records ensure
        that deterministic category/branch/percentile filtering can see every valid
        indexed option.
        """
        semantic = self.retrieve_multiple(
            queries=queries,
            top_k_per_query=top_k_per_query,
            final_limit=final_limit,
            document_type="cutoff",
            university=preferred_university,
        )
        exhaustive = self.store.list_cutoff_records(
            document_type="cutoff",
            university=preferred_university,
        )

        combined: dict[str, dict[str, Any]] = {}
        for result in semantic + exhaustive:
            key = self._result_key(result)
            existing = combined.get(key)
            if existing is None or float(result.get("similarity", 0.0) or 0.0) > float(existing.get("similarity", 0.0) or 0.0):
                combined[key] = result
        return list(combined.values())

    def filter_cutoff_candidates(
        self,
        results: list[dict[str, Any]],
        category: str,
        preferred_branches: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter retrieved cutoff chunks by structured metadata, never by student percentile."""
        category_code = re.sub(r"[^A-Z0-9]", "", str(category or "OPEN").upper())
        branch_terms = self._branch_terms(preferred_branches or [])
        output: list[dict[str, Any]] = []

        for result in results:
            metadata = result.get("metadata") or {}
            branch = str(metadata.get("branch") or metadata.get("course") or "")
            if branch_terms and not any(term in branch.casefold() for term in branch_terms):
                continue

            try:
                seat_cutoffs = json.loads(str(metadata.get("seat_cutoffs_json") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            matching = {
                seat: values
                for seat, values in seat_cutoffs.items()
                if self._seat_matches_category(seat, category_code)
            }
            if not matching:
                continue

            output.append({
                **result,
                "matching_seat_cutoffs": matching,
            })
        return output

    def build_compact_cutoff_context(
        self,
        results: list[dict[str, Any]],
        max_characters: int | None = None,
    ) -> str:
        """Build dense structured evidence so dozens of candidates fit in the LLM context."""
        limit = max_characters or self.settings.max_context_characters
        lines: list[str] = []
        used = 0
        for evidence_id, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            seats = result.get("matching_seat_cutoffs") or {}
            seat_text = "; ".join(
                f"{seat}={float(values.get('cutoff_percentile')):.7f}"
                for seat, values in seats.items()
                if values.get("cutoff_percentile") is not None
            )
            if not seat_text:
                continue
            line = (
                f"[Evidence {evidence_id}] College={metadata.get('college','')}; "
                f"Branch={metadata.get('branch') or metadata.get('course','')}; "
                f"Status={metadata.get('institute_status','Not available')}; "
                f"Allocation={metadata.get('seat_allocation','Not available')}; "
                f"Stage={metadata.get('stage','')}; Seats={seat_text}; "
                f"Page={metadata.get('page_number','unknown')}\n"
            )
            if used + len(line) > limit:
                break
            lines.append(line)
            used += len(line)
        return "".join(lines)

    @staticmethod
    def _seat_matches_category(seat_type: str, category: str) -> bool:
        code = re.sub(r"[^A-Z0-9]", "", str(seat_type or "").upper())
        cat = re.sub(r"[^A-Z0-9]", "", str(category or "OPEN").upper())
        if not code or any(token in code for token in ("DEF", "PWD", "TFWS")):
            return False
        tokens = {
            "OPEN": ("OPEN",), "OBC": ("OBC",), "SC": ("SC",),
            "ST": ("ST",), "EWS": ("EWS",), "SBC": ("SBC",),
            "VJ": ("VJ", "DT"), "NTA": ("NTA", "VJ"),
            "NTB": ("NTB",), "NTC": ("NTC",), "NTD": ("NTD",),
        }.get(cat, (cat,))
        return any(token and token in code for token in tokens)

    @staticmethod
    def _branch_terms(branches: list[str]) -> list[str]:
        aliases = {
            "computer engineering": ("computer engineering", "computer science", "cse", "computing"),
            "computer science": ("computer science", "computer engineering", "cse"),
            "computer science and engineering": ("computer science and engineering", "computer engineering", "computer science", "cse"),
            "cse": ("computer science and engineering", "computer engineering", "computer science", "cse"),
            "information technology": ("information technology",),
            "it": ("information technology",),
            "artificial intelligence": ("artificial intelligence", "ai and data science", "ai & data science"),
            "artificial intelligence & data science": ("artificial intelligence", "data science", "ai and data science", "ai & data science"),
            "artificial intelligence and data science": ("artificial intelligence", "data science", "ai and data science", "ai & data science"),
            "data science": ("data science", "artificial intelligence and data science", "artificial intelligence & data science"),
        }
        terms: list[str] = []
        for branch in branches:
            key = str(branch).strip().casefold()
            if not key or key == "any":
                continue
            terms.extend(aliases.get(key, (key,)))
        return list(dict.fromkeys(terms))

    def build_context(
        self,
        results: list[dict[str, Any]],
        max_characters: int | None = None,
    ) -> str:
        total_limit = (
            max_characters
            or self.settings.max_context_characters
        )

        per_document_limit = getattr(
            self.settings,
            "max_chars_per_document",
            900,
        )

        blocks: list[str] = []
        total_characters = 0

        for evidence_id, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}

            document_text = str(
                result.get("text")
                or result.get("document")
                or result.get("page_content")
                or ""
            ).strip()

            if not document_text:
                continue

            document_text = document_text[:per_document_limit]

            similarity = float(
                result.get("similarity", 0.0) or 0.0
            )

            source = (
                metadata.get("source_file")
                or metadata.get("source")
                or "unknown"
            )

            page = (
                metadata.get("page_number")
                or metadata.get("page")
                or "unknown"
            )

            block = "\n".join(
                [
                    f"[Evidence {evidence_id}]",
                    f"Similarity: {similarity:.4f}",
                    f"Source: {source}",
                    f"Page: {page}",
                    document_text,
                    "",
                ]
            )

            if total_characters + len(block) > total_limit:
                break

            blocks.append(block)
            total_characters += len(block)

        return "\n".join(blocks)

    @staticmethod
    def _clean_queries(queries: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        if not isinstance(queries, list):
            return cleaned

        for query in queries:
            if not isinstance(query, str):
                continue

            normalized = " ".join(query.strip().split())
            key = normalized.casefold()

            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)

        return cleaned

    @staticmethod
    def _result_key(result: dict[str, Any]) -> str:
        metadata = result.get("metadata") or {}

        key = "|".join(
            str(metadata.get(field, "")).strip().casefold()
            for field in (
                "source_file",
                "institute_code",
                "choice_code",
                "college",
                "branch",
                "seat_allocation",
                "stage",
                "seat_cutoffs_json",
            )
        )

        if key.replace("|", ""):
            return key

        return str(
            result.get("id")
            or result.get("text")
            or id(result)
        )