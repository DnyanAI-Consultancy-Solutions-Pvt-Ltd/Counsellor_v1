from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class TextChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _State:
    institute_code: str = ""
    college: str = ""
    choice_code: str = ""
    course: str = ""
    institute_status: str = ""
    seat_allocation: str = ""
    seat_columns: list[tuple[str, float]] = field(default_factory=list)


class DocumentChunker:
    COLLEGE_PATTERN = re.compile(r"^\s*(\d{4})\s*-\s*(.+?)\s*$")
    COURSE_PATTERN = re.compile(r"^\s*(\d{9}[A-Z]?)\s*-\s*(.+?)\s*$")
    STATUS_PATTERN = re.compile(r"^\s*Status:\s*(.+?)\s*$", re.I)
    STAGE_PATTERN = re.compile(r"^\s*(I{1,3}|IV|V)\b")
    PERCENTILE_PATTERN = re.compile(r"\((\d{1,3}(?:\.\d+)?)\)")
    NUMBER_PATTERN = re.compile(r"\b\d{1,7}\b")

    ALLOCATION_MARKERS = (
        "Seats Allotted",
        "State Level",
        "All India Seats",
        "Maharashtra State Seats",
        "Minority Seats",
        "Institute Level",
    )

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 180) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: Iterable[Any],
        base_metadata: dict[str, Any] | None = None,
        source_file: str | None = None,
        document_type: str | None = None,
        **_: Any,
    ) -> list[TextChunk]:
        base = dict(base_metadata or {})
        if source_file:
            base["source_file"] = source_file
        if document_type:
            base["document_type"] = document_type

        output: list[TextChunk] = []
        state = _State()

        for index, page in enumerate(pages, start=1):
            text, metadata = self._page_content(page, index)
            merged = {**base, **metadata}
            is_cutoff = (
                str(merged.get("document_type", "")).casefold() == "cutoff"
                or self._looks_like_cutoff(text)
            )

            if is_cutoff:
                parsed = self._parse_cap_page(text, merged, state)
                if parsed:
                    output.extend(parsed)
                    continue

            output.extend(self._generic_chunks(text, merged))
        return output

    def _parse_cap_page(
        self,
        text: str,
        metadata: dict[str, Any],
        state: _State,
    ) -> list[TextChunk]:
        lines = text.splitlines()
        records: list[TextChunk] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            normalized = " ".join(line.strip().split())

            match = self.COLLEGE_PATTERN.match(normalized)
            if match:
                state.institute_code, state.college = match.group(1), match.group(2)
                state.choice_code = state.course = state.institute_status = ""
                state.seat_allocation = ""
                state.seat_columns = []
                i += 1
                continue

            match = self.COURSE_PATTERN.match(normalized)
            if match:
                state.choice_code, state.course = match.group(1), match.group(2)
                state.seat_allocation = ""
                state.seat_columns = []
                i += 1
                continue

            match = self.STATUS_PATTERN.match(normalized)
            if match:
                state.institute_status = match.group(1).strip()
                i += 1
                continue

            if self._is_allocation_heading(normalized):
                state.seat_allocation = normalized
                i += 1
                continue

            if re.search(r"\bStage\b", line):
                state.seat_columns = self._extract_headers(line)
                i += 1
                continue

            stage_match = self.STAGE_PATTERN.match(line)
            if stage_match and state.seat_columns:
                stage = stage_match.group(1)
                rank_values = [
                    (int(m.group(0)), float(m.start()))
                    for m in self.NUMBER_PATTERN.finditer(line, stage_match.end())
                ]
                percentile_index = self._find_percentile_line(lines, i + 1)
                if percentile_index is not None:
                    percentile_values = [
                        (float(m.group(1)), float(m.start()))
                        for m in self.PERCENTILE_PATTERN.finditer(lines[percentile_index])
                        if 0 <= float(m.group(1)) <= 100
                    ]
                    records.extend(
                        self._build_records(
                            state,
                            metadata,
                            stage,
                            rank_values,
                            percentile_values,
                        )
                    )
                    i = percentile_index + 1
                    continue
            i += 1

        return records

    def _build_records(
        self,
        state: _State,
        metadata: dict[str, Any],
        stage: str,
        rank_values: list[tuple[int, float]],
        percentile_values: list[tuple[float, float]],
    ) -> list[TextChunk]:
        rank_map = self._map_to_columns(state.seat_columns, rank_values)
        percentile_map = self._map_to_columns(state.seat_columns, percentile_values)

        seat_cutoffs: dict[str, dict[str, Any]] = {}
        for seat_type, _ in state.seat_columns:
            cutoff = percentile_map.get(seat_type)
            if cutoff is None:
                continue

            cutoff = float(cutoff)
            if not 0 <= cutoff <= 100:
                continue

            seat_cutoffs[seat_type] = {
                "cutoff_percentile": cutoff,
                "merit_rank": rank_map.get(seat_type),
            }

        if not seat_cutoffs:
            return []

        cutoff_values = [v["cutoff_percentile"] for v in seat_cutoffs.values()]
        cutoff_lines = [
            (
                f"{seat_type}: Cutoff Percentile={values['cutoff_percentile']:.7f}, "
                f"Merit Rank={values['merit_rank'] if values['merit_rank'] is not None else 'Not available'}"
            )
            for seat_type, values in seat_cutoffs.items()
        ]

        record_metadata = {
            **metadata,
            "record_type": "cap_branch_cutoff",
            "institute_code": state.institute_code,
            "college": state.college,
            "choice_code": state.choice_code,
            "course": state.course,
            "branch": state.course,
            "institute_status": state.institute_status or "Not available",
            "seat_allocation": state.seat_allocation or "Not available",
            "stage": stage,
            "seat_types": ",".join(seat_cutoffs.keys()),
            "seat_type_count": len(seat_cutoffs),
            "minimum_cutoff": min(cutoff_values),
            "maximum_cutoff": max(cutoff_values),
            "seat_cutoffs_json": json.dumps(seat_cutoffs, ensure_ascii=False),
        }

        text = "\n".join([
            "Record Type: MHT-CET CAP Branch Cutoff",
            f"College: {state.college}",
            f"Institute Code: {state.institute_code}",
            f"Course: {state.course}",
            f"Choice Code: {state.choice_code}",
            f"Institute Status: {state.institute_status or 'Not available'}",
            f"Seat Allocation: {state.seat_allocation or 'Not available'}",
            f"Stage: {stage}",
            "Seat-Type Cutoffs:",
            *cutoff_lines,
            f"Source Page: {metadata.get('page_number', 'unknown')}",
        ])
        return [TextChunk(text=text, metadata=record_metadata)]

    @staticmethod
    def _map_to_columns(
        columns: list[tuple[str, float]],
        values: list[tuple[Any, float]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        used: set[str] = set()
        for value, position in values:
            candidates = [
                (abs(position - col_position), seat)
                for seat, col_position in columns
                if seat not in used
            ]
            if not candidates:
                break
            distance, seat = min(candidates, key=lambda item: item[0])
            if distance <= 45:
                result[seat] = value
                used.add(seat)
        return result

    @staticmethod
    def _extract_headers(line: str) -> list[tuple[str, float]]:
        headers: list[tuple[str, float]] = []
        for match in re.finditer(r"[A-Z][A-Z0-9]{2,15}", line):
            token = match.group(0)
            if token not in {"STAGE", "STATUS", "STATE", "LEVEL"}:
                headers.append((token, float(match.start())))
        return headers

    def _find_percentile_line(self, lines: list[str], start: int) -> int | None:
        for index in range(start, min(start + 5, len(lines))):
            if self.PERCENTILE_PATTERN.search(lines[index]):
                return index
            if self.STAGE_PATTERN.match(lines[index]):
                return None
        return None

    def _generic_chunks(self, text: str, metadata: dict[str, Any]) -> list[TextChunk]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(
                    TextChunk(
                        text=chunk,
                        metadata={
                            **metadata,
                            "record_type": "generic_text",
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    @classmethod
    def _is_allocation_heading(cls, line: str) -> bool:
        return any(marker.casefold() in line.casefold() for marker in cls.ALLOCATION_MARKERS)

    @staticmethod
    def _looks_like_cutoff(text: str) -> bool:
        lowered = text.casefold()
        return "cut off list" in lowered and "merit percentile" in lowered

    @staticmethod
    def _page_content(page: Any, default_page: int) -> tuple[str, dict[str, Any]]:
        if isinstance(page, dict):
            metadata = dict(page.get("metadata") or {})
            metadata.setdefault("page_number", page.get("page_number", default_page))
            return str(page.get("text") or ""), metadata

        metadata = dict(getattr(page, "metadata", {}) or {})
        metadata.setdefault("page_number", default_page)
        return str(getattr(page, "page_content", page) or ""), metadata
