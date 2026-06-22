"""Main RAG engine orchestrating retrieval and generation."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class RAGResult:
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    latency_ms: float
    chunks_used: int

class RAGEngine:
    def __init__(self, embedding_model="BAAI/bge-large-en-v1.5", 
                 reranker=None, chunk_size=512, chunk_overlap=64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = HybridRetriever(embedding_model, chunk_size)
        self.reranker = CrossEncoderReranker(reranker) if reranker else None
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.documents = []
        
    def ingest(self, path, format="auto"):
        path = Path(path)
        if path.is_file():
            self._ingest_file(path)
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    self._ingest_file(f)
                    
    def _ingest_file(self, path):
        text = self._read_file(path)
        chunks = self.chunker.chunk(text, source=str(path))
        self.retriever.index(chunks)
        self.documents.extend(chunks)
        
    def query(self, question, top_k=5, filter_metadata=None):
        import time
        start = time.monotonic()
        
        candidates = self.retriever.search(question, top_k=top_k * 3, filter=filter_metadata)
        
        if self.reranker:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        
        context = "\n\n".join([c.text for c in candidates])
        answer = self._generate(question, context)
        
        return RAGResult(
            answer=answer,
            citations=[{"source": c.source, "chunk_id": c.id, "score": c.score} for c in candidates],
            confidence=candidates[0].score if candidates else 0.0,
            latency_ms=(time.monotonic() - start) * 1000,
            chunks_used=len(candidates),
        )
        
    def _read_file(self, path):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
            
    def _generate(self, question, context):
        return f"Based on the provided context: {question}\n\nContext: {context[:200]}..."
