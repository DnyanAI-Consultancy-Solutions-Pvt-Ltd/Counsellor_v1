def split_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            cut = max(chunk.rfind(". "), chunk.rfind("; "))
            if cut > chunk_size // 2:
                end = start + cut + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
