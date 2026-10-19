**# pgvector Benchmark Results (Track A: DBMS)**



**Part of RAGScale: Benchmarking Vector Database Architectures for Cloud-Based Document Intelligence.**



**## 1. Setup**



**- Dataset: 100,000 arXiv papers (cs.AI, cs.CL, cs.LG), title + abstract**

**- Embeddings: all-MiniLM-L6-v2, 384 dimensions; distance metric: L2**

**- Database: PostgreSQL 18.6 + pgvector 0.8.6 on a local Windows laptop (about 7.9 GB RAM), default shared\_buffers = 128 MB unless stated**

**- Table sizes tested: 1k, 10k, 20k, 50k, 100k rows (nested subsets, ids 1 to n); the 100k table is 270 MB**



**## 2. Method**



**- 50 fixed queries (papers drawn from the 1k table, seed 0.42), identical in every tier; each query's own id is removed from its results**

**- Top-10 nearest neighbours; recall@10 is measured against exact search**

**- Latency is measured from Python around each query: one warm-up pass, then the median of 3 timed passes per query; averages and P95 are over the 50 queries**

**- Index settings: HNSW m=16, ef\_construction=64 with ef\_search 40 and 100; IVFFlat lists about sqrt(rows) with probes 10 and 30; maintenance\_work\_mem = 1 GB for index builds**

**- Exact baseline measured single-process (parallel workers off) in exact\_baseline.py**



**## 3. Results**



**### 3.1 Index tuning at 20k rows (first benchmark)**



**| Config | Build (s) | Avg (ms) | P95 (ms) | Recall@10 |**

**|---|---|---|---|---|**

**| No index (exact) | 0 | 279.22 | 590.18 | 1.000 |**

**| HNSW m=16, ef\_construction=64 | 28.2 | 2.82 | 3.90 | 0.994 |**

**| HNSW m=24, ef\_construction=100 | 34.3 | 5.44 | 10.74 | 1.000 |**

**| IVFFlat lists=50 | 4.3 | 7.84 | 10.93 | 0.990 |**

**| IVFFlat lists=100 | 4.2 | 4.74 | 7.51 | 0.954 |**

**| IVFFlat lists=200 | 6.3 | 3.04 | 4.79 | 0.930 |**



**In this early run the queries came from the table itself, so each query's own paper was among its 10 results and recall is slightly inflated. Section 3.2 removes this.**



**### 3.2 Scaling from 1k to 100k rows**



**Average latency in ms (recall@10 in brackets):**



**| Rows | Exact (single-process) | HNSW ef\_search=40 | IVFFlat probes=10 | IVFFlat probes=30 |**

**|---|---|---|---|---|**

**| 1k | 5.6 | 0.49 (1.000) | 0.47 (0.978) | 0.73 (1.000) |**

**| 10k | 182 | 2.65 (1.000) | 2.49 (0.930) | 4.88 (0.990) |**

**| 20k | 228 | 2.62 (0.998) | 3.02 (0.942) | 9.66 (0.988) |**

**| 50k | 673 | 2.48 (0.990) | 5.33 (0.920) | 15.5 (0.982) |**

**| 100k | 2391 | 5.57 (0.994) | 6.25 (0.930) | 23.2 (0.984) |**



**HNSW with ef\_search=100 reached recall 0.996 at 50k and 100k, at 4.02 ms and 19.05 ms.**



**Index build time (s):**



**| Rows | HNSW (m=16, ef\_c=64) | IVFFlat |**

**|---|---|---|**

**| 1k | 0.2 | 0.0 |**

**| 10k | 3.2 | 6.9 |**

**| 20k | 14.1 | 7.8 |**

**| 50k | 44.8 | 18.0 |**

**| 100k | 113.6 | 34.8 |**



**Charts (from scale\_results\_median.csv):**



**!\[Latency vs data size](charts/latency.png)**

**!\[Recall vs data size](charts/recall.png)**

**!\[Build time vs data size](charts/build\_time.png)**



**### 3.3 Query-plan analysis**



**- Without an index, a 20k-row search was a sequential scan plus sort (about 1,288 ms in the first EXPLAIN ANALYZE). With HNSW, PostgreSQL used an index scan on the same query (about 4.9 ms).**

**- At 100k rows PostgreSQL planned a parallel sequential scan for exact search (Gather Merge, 2 workers planned); at 50k it used a plain sequential scan. The exact baseline in 3.2 is therefore measured single-process so every size is timed the same way. With default settings, exact search at 100k took about 643 ms using parallel workers.**

**- The 100k table (270 MB) is larger than shared\_buffers (128 MB).**



**### 3.4 Memory experiment (shared\_buffers 128 MB vs 1 GB)**



**<!-- FILL AFTER compare\_buffers.py: paste the 50k and 100k comparison lines and one conclusion sentence -->**

**TODO**



**## 4. Findings**



**1. \*\*Indexing is essential.\*\* Exact search grows steadily with data size (5.6 ms at 1k to about 2.4 s at 100k, single-process), while HNSW stays between 0.5 and 5.6 ms. HNSW is about 70x faster than exact search at 10k and about 430x faster at 100k.**

**2. \*\*HNSW gives the best speed at high recall.\*\* With ef\_search=40, recall stays at 0.99 or higher at every size, with latency of a few milliseconds.**

**3. \*\*More aggressive HNSW build settings trade speed for accuracy.\*\* At 20k, m=24 and ef\_construction=100 raised recall from 0.994 to 1.000, but roughly doubled latency and lengthened the build.**

**4. \*\*IVFFlat trades recall for latency through probes.\*\* With probes=10, recall stays around 0.92 to 0.94 from 10k rows up. With probes=30 it recovers to 0.98 to 0.99, but then takes 4 to 6 times as long as HNSW at 50k and 100k.**

**5. \*\*IVFFlat builds faster.\*\* About 3 times faster at 50k and 100k (35 s vs 114 s at 100k), which matters when data changes often.**

**6. \*\*One unexplained jump.\*\* HNSW with ef\_search=100 went from 4.0 ms at 50k to 19 ms at 100k. The 100k table and index are larger than shared\_buffers, which section 3.4 tests.**



**## 5. Limitations**



**- Single laptop, local PostgreSQL, warm cache; not yet repeated on cloud hardware**

**- 50 queries, so P95 values are approximate; build times are single measurements and vary between runs (HNSW at 10k: 6.5 s in one run, 3.2 s in another)**

**- Tiers are nested subsets, so the paper mix shifts slightly with size**

**- Latency is measured from Python and includes small client overhead, applied equally to every configuration**

**- Only HNSW m=16 and ef\_construction=64 were tested across all tiers; other settings were compared at 20k only**



**## 6. Next steps**



**- Repeat the benchmark on AWS RDS PostgreSQL with pgvector (run from a client in the same region)**

**- Run the same queries against the dedicated vector database through the shared interface (common/backend\_interface.py), scored against common/ground\_truth.json**

**- Combine into the team's cost, latency, recall, and scaling comparison**



**## Files**



**benchmark\_indexes.py, benchmark\_scale.py, exact\_baseline.py, export\_ground\_truth.py, pgvector\_backend.py, test\_backend.py, make\_charts.py, compare\_buffers.py; raw data in scale\_results\_median.csv, scale\_results.csv (first noisier run), and scale\_results\_1gb.csv.**

