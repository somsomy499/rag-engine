"""Tests for RAG Engine."""
from rag_engine import RAGEngine

def test_init():
    engine = RAGEngine(chunk_size=256)
    assert engine.chunk_size == 256
    assert len(engine.documents) == 0
