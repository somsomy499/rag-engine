# Benchmarks

## Retrieval Quality

| Dataset | Recall@5 | Recall@10 | MRR@10 |
|---------|----------|-----------|--------|
| NQ | 0.82 | 0.88 | 0.71 |
| HotpotQA | 0.76 | 0.83 | 0.65 |
| MS MARCO | 0.85 | 0.91 | 0.78 |
| Feverous | 0.79 | 0.86 | 0.69 |

## End-to-End QA

| System | EM | F1 | Latency |
|--------|-----|-----|---------|
| BM25 + GPT-4 | 0.52 | 0.68 | 2.1s |
| Dense + GPT-4 | 0.61 | 0.75 | 1.8s |
| Hybrid + GPT-4 | 0.67 | 0.80 | 1.9s |
| Ours (full) | 0.71 | 0.84 | 2.0s |

## Latency Breakdown

| Component | Time |
|-----------|------|
| Query embedding | 15ms |
| Dense retrieval | 8ms |
| BM25 retrieval | 3ms |
| Reranking | 45ms |
| LLM generation | 800ms |
| **Total** | **~871ms** |
