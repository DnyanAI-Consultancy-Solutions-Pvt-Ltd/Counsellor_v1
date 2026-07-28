from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentLoader:
    """
    Loads documents into a common page structure.

    For PDFs, layout-preserving extraction is used whenever supported by
    pypdf. This preserves table column positions without embedding any
    counselling rules or college-specific logic.
    """

    ROUND_PATTERN = re.compile(r"\bCAP\s+Round\s+([IVX]+)\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(20\d{2})\s*[-–]\s*(\d{2,4})\b")

    def load(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(path)

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {extension}")

        if extension == ".pdf":
            return self._load_pdf(path)
        if extension == ".docx":
            return self._load_docx(path)
        return self._load_text(path)

    def _load_pdf(self, path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(path))
        pages: list[dict[str, Any]] = []

        detected_round = ""
        detected_year = ""

        for index, page in enumerate(reader.pages, start=1):
            text = self._extract_pdf_text(page)

            if not detected_round:
                match = self.ROUND_PATTERN.search(text)
                if match:
                    detected_round = match.group(1).upper()

            if not detected_year:
                match = self.YEAR_PATTERN.search(text)
                if match:
                    start_year = match.group(1)
                    end_year = match.group(2)
                    if len(end_year) == 2:
                        end_year = start_year[:2] + end_year
                    detected_year = f"{start_year}-{end_year}"

            pages.append(
                {
                    "text": text.rstrip(),
                    "page_number": index,
                    "metadata": {
                        "file_name": path.name,
                        "source_file": path.name,
                        "file_type": "pdf",
                        "extraction_mode": "layout",
                        "cap_round": detected_round,
                        "academic_year": detected_year,
                    },
                }
            )

        return pages

    @staticmethod
    def _extract_pdf_text(page: Any) -> str:
        try:
            text = page.extract_text(extraction_mode="layout")
        except (TypeError, ValueError, NotImplementedError):
            text = page.extract_text()

        return text or ""

    def _load_docx(self, path: Path) -> list[dict[str, Any]]:
        document = Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

        return [
            {
                "text": "\n".join(paragraphs),
                "page_number": 1,
                "metadata": {
                    "file_name": path.name,
                    "source_file": path.name,
                    "file_type": "docx",
                    "extraction_mode": "paragraph",
                },
            }
        ]

    def _load_text(self, path: Path) -> list[dict[str, Any]]:
        return [
            {
                "text": path.read_text(encoding="utf-8", errors="ignore"),
                "page_number": 1,
                "metadata": {
                    "file_name": path.name,
                    "source_file": path.name,
                    "file_type": path.suffix.lower()[1:],
                    "extraction_mode": "plain_text",
                },
            }
        ]
