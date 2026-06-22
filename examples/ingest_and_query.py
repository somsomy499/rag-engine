"""Full example: ingest documents and query."""
from rag_engine import RAGEngine

# Initialize engine
engine = RAGEngine(
    embedding_model="BAAI/bge-large-en-v1.5",
    reranker="BAAI/bge-reranker-v2-m3",
    chunk_size=512,
    chunk_overlap=64,
)

# Ingest documents
print("Ingesting documents...")
engine.ingest("./docs/", format="auto")
print(f"Indexed {len(engine.documents)} chunks")

# Query
result = engine.query(
    "What is the main architecture pattern?",
    top_k=5,
)
print(f"\nAnswer: {result.answer}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Sources: {len(result.citations)}")
print(f"Latency: {result.latency_ms:.0f}ms")

for cite in result.citations[:3]:
    print(f"  - {cite['source']} (chunk {cite['chunk_id']}, score {cite['score']:.3f})")
