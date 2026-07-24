from rag.chunking import split_text
def test_split_text():
    chunks=split_text("A sentence. "*500,chunk_size=200,overlap=20)
    assert len(chunks)>1 and all(chunks)
