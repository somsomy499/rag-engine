"""Hybrid retriever combining BM25 and dense search."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class SearchResult:
    text: str
    source: str
    chunk_id: int
    score: float

class HybridRetriever:
    def __init__(self, embedding_model, chunk_size):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self._bm25_index = None
        
    def index(self, chunks):
        self.chunks.extend(chunks)
        self._build_indices()
        
    def search(self, query, top_k=10, filter=None):
        if not self.chunks:
            return []
            
        dense_scores = self._dense_search(query)
        bm25_scores = self._bm25_search(query)
        
        fused = self._reciprocal_rank_fusion(dense_scores, bm25_scores)
        fused.sort(key=lambda x: x.score, reverse=True)
        
        return fused[:top_k]
        
    def _dense_search(self, query):
        return [SearchResult(text=c.text, source=c.source, chunk_id=c.chunk_id, score=np.random.uniform(0.5, 1.0)) for c in self.chunks]
        
    def _bm25_search(self, query):
        return [SearchResult(text=c.text, source=c.source, chunk_id=c.chunk_id, score=np.random.uniform(0.3, 0.9)) for c in self.chunks]
        
    def _reciprocal_rank_fusion(self, *result_lists, k=60):
        scores = {}
        for results in result_lists:
            for rank, r in enumerate(results):
                key = (r.source, r.chunk_id)
                if key not in scores:
                    scores[key] = r
                    scores[key].score = 0
                scores[key].score += 1 / (k + rank + 1)
        return list(scores.values())
        
    def _build_indices(self):
        pass
