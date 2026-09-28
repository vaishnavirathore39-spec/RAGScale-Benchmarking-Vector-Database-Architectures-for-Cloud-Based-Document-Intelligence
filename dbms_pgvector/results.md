**# pgvector Index Benchmark (Track A)**



**Setup: PostgreSQL 18.6 + pgvector 0.8.6 (local), 20,000 arXiv papers**

**(cs.AI, cs.CL, cs.LG), 384-dim embeddings (all-MiniLM-L6-v2),**

**top-5 nearest neighbours by L2 distance, single query, EXPLAIN ANALYZE.**



**| Method | Execution time |**

**|---|---|**

**| No index (sequential scan) | 1288.2 ms |**

**| HNSW default (m=16, ef\_construction=64) | 4.55 ms |**

**| HNSW tuned (m=24, ef\_construction=100) | 9.61 ms |**

**| IVFFlat (lists=100) | 4.90 ms |**



**Notes:**

**- Single run per config, so treat small differences as noise.**

**- Tuned HNSW ran on a cold cache and used more buffers. Needs a repeat run.**

**- Recall not measured yet.**



**TODO: repeat runs, IVFFlat lists=50/200, scale tests (1k/10k/100k), AWS deployment.**



