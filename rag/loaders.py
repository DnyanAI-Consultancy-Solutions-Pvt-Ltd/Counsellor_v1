from pathlib import Path
import json
import pandas as pd
from pypdf import PdfReader
from docx import Document

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls", ".json"}


def load_file(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")
    out: list[dict] = []
    if ext == ".pdf":
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                out.append({"text": text, "page": i, "section": ""})
    elif ext == ".docx":
        text = "\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
        out = [{"text": text, "page": 1, "section": ""}]
    elif ext in {".txt", ".md"}:
        out = [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page": 1, "section": ""}]
    elif ext == ".csv":
        df = pd.read_csv(path)
        out = [{"text": df.to_csv(index=False), "page": 1, "section": "table"}]
    elif ext in {".xlsx", ".xls"}:
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            out.append({"text": df.to_csv(index=False), "page": 1, "section": str(sheet)})
    elif ext == ".json":
        out = [{"text": json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), "page": 1, "section": "json"}]
    return out
