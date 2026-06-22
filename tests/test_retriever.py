"""Tests for hybrid retriever."""
from rag_engine.retriever import HybridRetriever, SearchResult

def test_empty_search():
    retriever = HybridRetriever("fake-model", 512)
    results = retriever.search("query")
    assert results == []

def test_search_after_index():
    from rag_engine.chunker import Chunk
    retriever = HybridRetriever("fake-model", 512)
    chunks = [Chunk(text="hello world", source="test", chunk_id=0)]
    retriever.index(chunks)
    results = retriever.search("hello", top_k=1)
    assert len(results) == 1
    assert results[0].text == "hello world"
