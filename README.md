# RAG Counsellor V2 Complete

This package includes the previously missing `rag/embeddings.py` and a complete runnable backend/UI.

## Architecture

- CAP-aware layout loader and structured cutoff chunker
- Sentence Transformer embeddings
- Persistent ChromaDB
- LLM-created retrieval plan
- Counsellor Agent
- Feedback Agent
- Excel export and session storage
- FastAPI backend
- Streamlit UI

Python does not contain fixed college names, branch priorities, or fixed Dream/Target/Safe percentile bands. The parser only interprets the official document structure. The LLM makes counselling decisions from retrieved evidence.

## Windows setup

```powershell
cd C:\RAG_Counsellor_V2_Complete
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Add your Groq API key to `.env`.

## Start backend

```powershell
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Start UI

In a second terminal:

```powershell
cd C:\RAG_Counsellor_V2_Complete
venv\Scripts\activate
streamlit run app.py
```

## First run

1. Use `DELETE /knowledge-base` to clear old vectors.
2. Upload the CAP cutoff PDF with `document_type=cutoff`.
3. Confirm `structured_cutoff_records` is greater than zero.
4. Call `/counsel` or use the Streamlit UI.

The first embedding run downloads the configured Sentence Transformer model and can take several minutes.
