import json
from pathlib import Path
from threading import RLock
from config.settings import get_settings
from models.schemas import StudentProfile, Recommendation


class SessionStore:
    def __init__(self, path: Path | None = None):
        self.path = path or get_settings().memory_file
        self.lock = RLock()
        if not self.path.exists():
            self._write({})

    def _read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write(self, data):
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, session_id: str) -> dict:
        with self.lock:
            return self._read().get(session_id, {
                "profile": {}, "history": [], "recommendations": [], "workbook_version": 0
            })

    def save(
        self,
        session_id: str,
        profile: StudentProfile,
        user_message: str,
        answer: str,
        recommendations: list[Recommendation],
        workbook_version: int,
    ) -> None:
        with self.lock:
            data = self._read()
            entry = data.setdefault(session_id, {})
            entry["profile"] = profile.model_dump()
            entry.setdefault("history", []).append({"user": user_message, "assistant": answer})
            entry["history"] = entry["history"][-12:]
            entry["recommendations"] = [row.model_dump() for row in recommendations]
            entry["workbook_version"] = workbook_version
            self._write(data)

    def clear(self, session_id: str) -> None:
        with self.lock:
            data = self._read()
            data.pop(session_id, None)
            self._write(data)
