"""Semantic chunking with overlap optimization."""
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict = None

class SemanticChunker:
    def __init__(self, chunk_size=512, overlap=64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current = ""
        chunk_id = 0
        
        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(Chunk(text=current.strip(), source=source, chunk_id=chunk_id))
                chunk_id += 1
                overlap_text = current[-self.overlap:] if self.overlap else ""
                current = overlap_text + " " + sent
            else:
                current += " " + sent if current else sent
                
        if current.strip():
            chunks.append(Chunk(text=current.strip(), source=source, chunk_id=chunk_id))
            
        return chunks
        
    def _split_sentences(self, text):
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
