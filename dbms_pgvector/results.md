**# pgvector Index Benchmark (Track A)**



**Setup: PostgreSQL 18.6 + pgvector 0.8.6 (local), 20,000 arXiv papers**

**(cs.AI, cs.CL, cs.LG), 384-dim embeddings (all-MiniLM-L6-v2).**

**50 random queries, top-10 by L2 distance, latency measured from Python.**

**Recall@10 is measured against exact (no-index) results.**

**HNSW ef\_search=40, IVFFlat probes=10.**



**| Config | Build (s) | Avg (ms) | P95 (ms) | Recall@10 |**

**|---|---|---|---|---|**

**| No index (exact) | 0 | 279.22 | 590.18 | 1.000 |**

**| HNSW m=16, ef\_c=64 | 28.2 | 2.82 | 3.90 | 0.994 |**

**| HNSW m=24, ef\_c=100 | 34.3 | 5.44 | 10.74 | 1.000 |**

**| IVFFlat lists=50 | 4.3 | 7.84 | 10.93 | 0.990 |**

**| IVFFlat lists=100 | 4.2 | 4.74 | 7.51 | 0.954 |**

**| IVFFlat lists=200 | 6.3 | 3.04 | 4.79 | 0.930 |**



**Findings:**

**- Default HNSW gives the best speed/recall balance at this scale.**

**- Higher HNSW m/ef\_construction raised recall to 1.0 but doubled latency.**

**- IVFFlat builds \~7x faster; more lists = faster but lower recall at fixed probes.**



**Caveats:**

**- Queries are drawn from the table, so each query's nearest neighbour is itself (slightly inflates recall).**

**- IVFFlat probes fixed at 10; a probes sweep is still to do.**

**- Small sample (50 queries); P95 is approximate.**



**TODO: probes sweep, scale tests (1k/10k/100k), AWS deployment.**

