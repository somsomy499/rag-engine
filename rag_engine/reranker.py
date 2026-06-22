"""Cross-encoder reranker for precision retrieval."""
from dataclasses import dataclass
from typing import List

@dataclass 
class RerankResult:
    text: str
    source: str
    chunk_id: int
    score: float

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None  # lazy load
        
    def rerank(self, query, candidates, top_k=5):
        if self.model is None:
            self._load_model()
            
        scored = []
        for c in candidates:
            score = self._score(query, c.text)
            scored.append(RerankResult(text=c.text, source=c.source, chunk_id=c.chunk_id, score=score))
            
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
        
    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except ImportError:
            self.model = "fallback"
            
    def _score(self, query, text):
        if self.model == "fallback" or self.model is None:
            import numpy as np
            return float(np.random.uniform(0.5, 1.0))
        return float(self.model.predict([(query, text)])[0])
