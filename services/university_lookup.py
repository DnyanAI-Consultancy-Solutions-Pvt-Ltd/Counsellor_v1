from __future__ import annotations

from typing import Any

from rag.retriever import Retriever
from services.llm_service import LLMService, get_llm_service


class UniversityLookupService:
    """Resolve a university request against indexed general-directory evidence.

    This service never hardcodes university-to-college mappings. The LLM extracts
    the requested university from the user's natural-language request, then the
    retriever searches document_type="general" chunks. College names are extracted
    only from that retrieved evidence.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm_service or get_llm_service()

    def resolve(
        self,
        user_request: str | None,
    ) -> dict[str, Any]:
        request_text = str(user_request or "").strip()
        if not request_text:
            return self._empty()

        extracted = self.llm.generate_json(
            system_prompt="""
You extract an explicit university preference from an MHT-CET engineering
counselling request.

Return JSON only:
{
  "preferred_university": null
}

Rules:
1. Return a university only when the user is asking to restrict, prefer, or obtain
   results according to a university/university region/affiliation.
2. Do not invent a university.
3. If the user does not express a university preference, return null.
4. Preserve the name from the request when possible. Common abbreviations may be
   expanded only when you are confident (for example SPPU / Pune University).
""",
            user_prompt=f"USER REQUEST:\n{request_text}",
            temperature=0.0,
            max_tokens=250,
        )

        preferred_university = str(
            (extracted or {}).get("preferred_university") or ""
        ).strip()

        if not preferred_university:
            return self._empty()

        queries = self._queries(preferred_university, request_text)

        evidence = self.retriever.retrieve_multiple(
            queries=queries,
            top_k_per_query=20,
            final_limit=40,
            document_type="general",
        )

        clean_evidence = [
            item for item in evidence
            if str(item.get("text") or "").strip()
        ]

        if not clean_evidence:
            return {
                "preferred_university": preferred_university,
                "colleges": [],
                "evidence_count": 0,
                "source_files": [],
                "status": "university_found_but_no_general_evidence",
            }

        # Include source boundaries so the model can reason about directory sections
        # without mixing unrelated university records.
        blocks: list[str] = []
        source_files: list[str] = []
        for index, item in enumerate(clean_evidence, start=1):
            metadata = item.get("metadata") or {}
            source_file = str(metadata.get("source_file") or "unknown")
            page_number = metadata.get("page_number", "unknown")
            if source_file not in source_files:
                source_files.append(source_file)
            blocks.append(
                f"[EVIDENCE {index} | SOURCE={source_file} | PAGE={page_number}]\n"
                f"{str(item.get('text') or '').strip()}"
            )

        context = "\n\n".join(blocks)

        extracted_colleges = self.llm.generate_json(
            system_prompt="""
You are a grounded university-directory extraction tool.

Given:
- one requested university, and
- retrieved text from indexed engineering-college directory documents,

return ONLY colleges that the evidence places under, affiliates with, or groups
inside the requested university/university section.

Return JSON only:
{
  "colleges": []
}

Strict rules:
1. Use only the retrieved evidence below.
2. Never invent a college.
3. Do not include colleges from another university section.
4. If a retrieved chunk contains multiple university sections, use the visible
   headings/boundaries to determine which college rows belong to the requested one.
5. Prefer the full college name exactly as written in evidence.
6. Do not return district names, university names, headings, statuses or locations
   as colleges.
7. Remove duplicates.
8. If the evidence is insufficient or ambiguous, omit the uncertain college.
""",
            user_prompt=(
                f"REQUESTED UNIVERSITY:\n{preferred_university}\n\n"
                f"RETRIEVED GENERAL-DOCUMENT EVIDENCE:\n{context}"
            ),
            temperature=0.0,
            max_tokens=1600,
        )

        colleges_raw = (extracted_colleges or {}).get("colleges") or []
        if not isinstance(colleges_raw, list):
            colleges_raw = []

        colleges: list[str] = []
        seen: set[str] = set()
        for item in colleges_raw:
            name = " ".join(str(item or "").split())
            key = name.casefold()
            if name and key not in seen:
                seen.add(key)
                colleges.append(name)

        return {
            "preferred_university": preferred_university,
            "colleges": colleges,
            "evidence_count": len(clean_evidence),
            "source_files": source_files,
            "status": "resolved" if colleges else "university_found_but_no_colleges_extracted",
        }

    @staticmethod
    def _queries(university: str, request_text: str) -> list[str]:
        queries = [
            university,
            f"{university} engineering colleges",
            f"{university} affiliated engineering colleges",
            f"{university} university region college directory",
        ]
        if request_text and request_text.casefold() != university.casefold():
            queries.append(request_text)

        output: list[str] = []
        seen: set[str] = set()
        for query in queries:
            clean = " ".join(str(query).split())
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                output.append(clean)
        return output

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "preferred_university": None,
            "colleges": [],
            "evidence_count": 0,
            "source_files": [],
            "status": "no_university_requested",
        }


def get_university_lookup_service() -> UniversityLookupService:
    return UniversityLookupService()