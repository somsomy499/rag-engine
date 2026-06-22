"""Embedding providers abstraction."""
from typing import List, Protocol
import numpy as np

class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        
    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self._load()
        return self.model.encode(texts)
    
    def _load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)

class OpenAIEmbedding:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        
    def embed(self, texts: List[str]) -> np.ndarray:
        import openai
        client = openai.OpenAI()
        response = client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in response.data])

class FakeEmbedding:
    """For testing — returns random vectors."""
    def __init__(self, dim: int = 384):
        self.dim = dim
        
    def embed(self, texts: List[str]) -> np.ndarray:
        return np.random.randn(len(texts), self.dim).astype(np.float32)
