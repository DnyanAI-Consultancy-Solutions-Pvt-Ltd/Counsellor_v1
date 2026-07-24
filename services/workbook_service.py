from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from config.settings import get_settings
from models.schemas import Recommendation


COLUMNS = [
    "Preference No.", "Institute Code", "College", "City", "Course Code", "Course",
    "Seat Type", "Cutoff Percentile", "Match Category", "Reasoning Logic",
    "Source Filename", "Source Page",
]


class WorkbookService:
    def __init__(self) -> None:
        self.base_dir = get_settings().sessions_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        path = self.base_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workbook_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "cutoff_list.xlsx"

    @staticmethod
    def to_frame(rows: list[Recommendation]) -> pd.DataFrame:
        records = []
        for index, row in enumerate(rows, start=1):
            records.append({
                "Preference No.": index,
                "Institute Code": row.institute_code,
                "College": row.college,
                "City": row.city,
                "Course Code": row.course_code,
                "Course": row.course,
                "Seat Type": row.seat_type,
                "Cutoff Percentile": row.cutoff_percentile,
                "Match Category": row.match_category,
                "Reasoning Logic": row.reasoning_logic,
                "Source Filename": row.source_filename,
                "Source Page": row.source_page,
            })
        return pd.DataFrame(records, columns=COLUMNS)

    def save_atomic(self, session_id: str, rows: list[Recommendation]) -> Path:
        target = self.workbook_path(session_id)
        frame = self.to_frame(rows)
        with NamedTemporaryFile(suffix=".xlsx", dir=target.parent, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
                frame.to_excel(writer, index=False, sheet_name="Preference Sheet")
                sheet = writer.sheets["Preference Sheet"]
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for column_cells in sheet.columns:
                    width = min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 70)
                    sheet.column_dimensions[column_cells[0].column_letter].width = width
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return target
