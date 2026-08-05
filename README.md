---
title: MHT-CET Agentic RAG Counsellor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎓 MHT-CET Agentic RAG Counsellor V3

An Agentic Retrieval-Augmented Generation (RAG) application for MHT-CET Engineering CAP counselling.

## Features

- Multi-Agent Architecture
- AI Counsellor Agent
- AI Feedback Agent
- ChromaDB Vector Database
- LangGraph Workflow
- Streamlit Frontend
- FastAPI Backend
- Groq LLM
- Excel Recommendation Generation
- Interactive Feedback Chat
- Dynamic College Addition & Removal
- Dream / Target / Safe Classification
- Branch-wise Recommendation
- College-wise Recommendation
- Semantic Search over CAP PDFs

---

## Technology Stack

- Python 3.11
- Streamlit
- FastAPI
- LangGraph
- ChromaDB
- Sentence Transformers
- Groq API
- Pandas
- OpenPyXL
- PyMuPDF

---

## Project Workflow

1. Upload CAP Cutoff PDFs
2. Parse PDF Documents
3. Create Chunks
4. Generate Embeddings
5. Store in ChromaDB
6. Student enters profile
7. Counsellor Agent generates recommendations
8. Feedback Agent refines recommendations
9. Excel report generated

---

## Agent Workflow

### Counsellor Agent

- Retrieves cutoff information from ChromaDB
- Filters by
  - Percentile
  - Category
  - Gender
  - Preferred Branch
  - Preferred Location
- Classifies colleges into
  - Dream
  - Target
  - Safe
- Generates ranked recommendations

### Feedback Agent

- Understands natural language requests
- Adds colleges
- Removes colleges
- Suggests better branches
- Suggests better colleges
- Explains cutoff differences
- Requests confirmation before forcing high-cutoff colleges
- Updates recommendation list without regenerating the entire counselling result

---

## Deployment

This project is deployed as a Docker Space on Hugging Face.

---

## Author

Developed by DnyanAI