from __future__ import annotations

import json

from agents.prompts import (
    COUNSELLOR_ROUTING_PROMPT,
    COUNSELLOR_RETRIEVAL_PROMPT,
    COUNSELLOR_GENERATION_PROMPT,
    COUNSELLOR_EXPANSION_PROMPT,
)
from config.settings import get_settings
from models.llm import get_llm
from models.schemas import (
    AgentRoute,
    Recommendation,
    RecommendationBatch,
    RetrievalPlan,
    StudentProfile,
)
from rag.store import KnowledgeBase
from services.recommendation_service import deduplicate, unique_college_count


def _compact_evidence(evidence: list[dict]) -> list[dict]:
    return [
        {
            "source_filename": item["metadata"].get("filename", ""),
            "source_page": item["metadata"].get("page"),
            "document_type": item["metadata"].get("document_type", ""),
            "score": item.get("score"),
            "text": item["text"][:1800],
        }
        for item in evidence
    ]


def decide_action(
    message: str,
    profile: StudentProfile,
    history: list[dict],
    existing_recommendations: list[Recommendation],
) -> AgentRoute:
    model = get_llm(fast=True, max_tokens=1200).with_structured_output(AgentRoute)
    prompt = f"""{COUNSELLOR_ROUTING_PROMPT}

CURRENT PROFILE:
{profile.model_dump_json()}

EXISTING PREFERENCE ROW COUNT:
{len(existing_recommendations)}

RECENT HISTORY:
{json.dumps(history[-6:], ensure_ascii=False)}

STUDENT MESSAGE:
{message}
"""
    return model.invoke(prompt)


def generate_initial(
    message: str,
    profile: StudentProfile,
    history: list[dict],
) -> tuple[list[Recommendation], str, str, list[dict], list[str]]:
    settings = get_settings()
    kb = KnowledgeBase()
    documents = kb.list_documents()
    if not documents:
        return [], "No cutoff documents are indexed. Upload the official CAP cutoff PDF in the Admin Knowledge Base first.", "Low", [], ["Counsellor Agent found no indexed cutoff evidence"]

    planner = get_llm(fast=True, max_tokens=1600).with_structured_output(RetrievalPlan)
    plan = planner.invoke(f"""{COUNSELLOR_RETRIEVAL_PROMPT}

INDEXED DOCUMENTS:
{json.dumps(documents, ensure_ascii=False)}

PROFILE:
{profile.model_dump_json()}

RECENT HISTORY:
{json.dumps(history[-4:], ensure_ascii=False)}

STUDENT REQUEST:
{message}
""")
    evidence = kb.search_many(plan.search_queries, document_types=plan.document_types or None)

    generator = get_llm(max_tokens=10000, temperature=0.05).with_structured_output(RecommendationBatch)
    batch = generator.invoke(f"""{COUNSELLOR_GENERATION_PROMPT.format(minimum_count=settings.minimum_recommendations, desired_count=settings.desired_recommendations)}

PROFILE:
{profile.model_dump_json()}

STUDENT REQUEST:
{message}

RETRIEVED CUTOFF EVIDENCE:
{json.dumps(_compact_evidence(evidence), ensure_ascii=False)}
""")
    rows = deduplicate(batch.recommendations)

    if unique_college_count(rows) < settings.minimum_recommendations:
        expansion_queries = list(plan.search_queries) + [
            f"MHT CET cutoff {profile.category or ''} {' '.join(profile.preferred_branches)} all institutes all rounds",
            f"engineering institute course cutoff percentile {profile.mht_cet_percentile} Maharashtra",
        ]
        extra_evidence = kb.search_many(
            expansion_queries,
            document_types=plan.document_types or None,
            n_results_per_query=settings.retrieval_results_per_query + 6,
            max_chunks=settings.max_evidence_chunks + 60,
        )
        expansion = generator.invoke(f"""{COUNSELLOR_EXPANSION_PROMPT}

MINIMUM REQUIRED: {settings.minimum_recommendations}
DESIRED: {settings.desired_recommendations}
PROFILE:
{profile.model_dump_json()}
CURRENT RECOMMENDATIONS:
{json.dumps([row.model_dump() for row in rows], ensure_ascii=False)}
ADDITIONAL EVIDENCE:
{json.dumps(_compact_evidence(extra_evidence), ensure_ascii=False)}
""")
        rows = deduplicate(rows + expansion.recommendations)
        evidence = extra_evidence
        if unique_college_count(rows) >= settings.desired_recommendations:
            selected, seen_colleges = [], set()
            for row in rows:
                college_key = row.college.strip().lower()
                if college_key in seen_colleges:
                    continue
                seen_colleges.add(college_key)
                selected.append(row)
                if len(selected) >= settings.desired_recommendations:
                    break
            rows = selected
        batch.counselling_summary = expansion.counselling_summary or batch.counselling_summary
        batch.confidence = expansion.confidence

    answer = batch.counselling_summary
    if unique_college_count(rows) < settings.minimum_recommendations:
        answer += f"\n\nThe indexed evidence supported {unique_college_count(rows)} distinct grounded colleges. Upload additional cutoff rounds or institute data to expand the list beyond this count."
    else:
        answer += f"\n\nGenerated {unique_college_count(rows)} distinct colleges across Dream, Target and Safe zones."

    trace = [
        "Counsellor Agent interpreted the initial counselling request",
        f"Counsellor Agent generated {len(plan.search_queries)} semantic retrieval queries",
        f"Counsellor Agent reviewed {len(evidence)} grounded cutoff chunks",
        f"Counsellor Agent produced {len(rows)} rows covering {unique_college_count(rows)} distinct colleges",
    ]
    return rows, answer, batch.confidence, evidence, trace
