from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class SessionService:
    def __init__(self, base_dir: str | Path = "storage/outputs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, payload: dict[str, Any]) -> str:
        session_id = str(uuid.uuid4())
        self.save(session_id, payload)
        return session_id

    def save(self, session_id: str, payload: dict[str, Any]) -> None:
        self._json_path(session_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._json_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown session: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def workbook_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.xlsx"

    def _json_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"
