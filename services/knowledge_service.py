from pathlib import Path
import shutil, uuid
from fastapi import UploadFile
from config.settings import get_settings
from rag.store import KnowledgeBase

class KnowledgeService:
    def __init__(self): self.kb=KnowledgeBase(); self.settings=get_settings()
    def save_and_index(self, upload: UploadFile) -> dict:
        safe=Path(upload.filename or "document").name; tmp=self.settings.uploads_dir/f"{uuid.uuid4().hex}_{safe}"
        with tmp.open("wb") as f: shutil.copyfileobj(upload.file,f)
        stable=self.settings.uploads_dir/safe; shutil.copy2(tmp,stable); tmp.unlink(missing_ok=True)
        return self.kb.index_file(stable)
