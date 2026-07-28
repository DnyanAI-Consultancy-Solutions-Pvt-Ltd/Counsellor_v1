RAG COUNSELLOR V2 PRODUCTION UPDATE
===================================

FILES TO REPLACE
----------------
rag/loader.py
rag/chunker.py
rag/store.py
rag/retriever.py
agents/counsellor.py
config/settings.py

WHY THIS IS NOT COUNSELLING HARDCODING
--------------------------------------
The parser contains only official document-structure rules:
- text in a table heading identifies the institute/course/seat column
- the value outside brackets is Merit Rank
- the value inside brackets is Merit Percentile
- table column positions map a seat type to its own official values

No college name, preferred branch, percentile band, Dream/Target/Safe rule,
location priority or admission choice is hardcoded in Python.

The LLM still:
- plans searches
- understands the profile
- selects options
- ranks choices
- assigns Dream/Target/Safe
- writes counselling reasons
- applies user feedback

REQUIRED STEPS
--------------
1. Stop FastAPI.
2. Back up the six existing files.
3. Replace them with the files from this package.
4. Delete/reset the old Chroma collection.
5. Restart FastAPI.
6. Upload 2023ENGG_CAP1_CutOff.pdf with document_type="cutoff".
7. Confirm upload output shows structured_cutoff_records greater than zero.
8. Run counselling again.

IMPORTANT
---------
Old Chroma vectors must be deleted. They contain malformed raw chunks and
cannot be repaired by replacing source code.

Recommended .env values:

MAX_SEARCH_QUERIES=10
RESULTS_PER_QUERY=25
MAX_CONTEXT_CHARACTERS=70000
RECOMMENDATION_COUNT=30

Do not upload a Seat Matrix PDF as document_type="cutoff". It contains seat
availability rather than historical cutoff percentile evidence.
