"""Tests for semantic chunker."""
from rag_engine.chunker import SemanticChunker, Chunk

def test_basic_chunking():
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunks = chunker.chunk(text, source="test.txt")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)

def test_chunk_ids():
    chunker = SemanticChunker(chunk_size=50, overlap=0)
    text = "A. " * 20
    chunks = chunker.chunk(text)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))

def test_empty_text():
    chunker = SemanticChunker()
    chunks = chunker.chunk("")
    assert chunks == []
