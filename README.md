# MHT-CET CAP Round AI Counsellor — Two-Agent Architecture

This version uses exactly two LangGraph agents:

1. **Counsellor Agent** — interprets the candidate profile, autonomously plans ChromaDB retrieval, generates the initial preference list, and creates `cutoff_list.xlsx`.
2. **Feedback / Re-ranker Agent** — receives natural-language modifications, reads the existing session recommendations, retrieves missing cutoff evidence when required, and updates `cutoff_list.xlsx` in place.

There is no keyword router and no fixed percentile-zone rule. The Counsellor Agent is the graph entry point and decides from conversation meaning whether to generate, clarify, answer, or hand off to the Feedback Agent. The minimum row count is a configurable product constraint (`MINIMUM_RECOMMENDATIONS=30`), not a hardcoded admissions rule.

## Run

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add GROQ_API_KEY and ADMIN_KEY to .env
python -m uvicorn api:app --reload
```

In another terminal:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Upload official cutoff PDFs from **Admin Knowledge Base**, then generate the first preference sheet from **Student Counsellor**. Follow-up prompts such as “Add VJTI, keep sorted” are handled by the Feedback Agent and update the same workbook.
