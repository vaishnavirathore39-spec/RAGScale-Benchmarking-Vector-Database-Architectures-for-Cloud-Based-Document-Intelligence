**# dbms\_pgvector (Track A: in-database vector search)**



**Benchmarks pgvector (HNSW and IVFFlat) on PostgreSQL 18 using 100k arXiv paper embeddings. Results and analysis are in `results.md`.**



**## Files**



**| File | Purpose |**

**|---|---|**

**| `benchmark\_indexes.py` | First tuning run: HNSW and IVFFlat settings compared on 20k rows |**

**| `benchmark\_scale.py` | Main benchmark: HNSW vs IVFFlat vs exact across 1k to 100k rows (median of 3 passes, recall@10) |**

**| `exact\_baseline.py` | Single-process exact-search timing for each size |**

**| `export\_ground\_truth.py` | Writes `../common/ground\_truth.json` (shared queries and exact top-10 ids) |**

**| `pgvector\_backend.py` | pgvector implementation of `../common/backend\_interface.py` |**

**| `test\_backend.py` | Checks the backend against the ground truth |**

**| `make\_charts.py` | Charts from `scale\_results\_median.csv` into `charts/` |**

**| `compare\_buffers.py` | Compares runs with shared\_buffers 128 MB vs 1 GB |**

**| `scale\_results\_\*.csv` | Raw benchmark output |**

**| `results.md` | Results, findings, limitations |**



**## Setup**



**1. Install PostgreSQL 18 and pgvector, then run `CREATE EXTENSION vector;`**

**2. Install Python packages: `pip install psycopg2-binary sentence-transformers matplotlib`**

**3. Data: download the arXiv dataset from Kaggle, run `data/filter.py` (100,000 records), then generate embeddings with `rag\_pipeline/embed.py` (first 20k) and `rag\_pipeline/embed\_extra.py` (rest), or download the embedding files from the team Drive folder. The embedding files are gitignored (too large for GitHub).**

**4. Create the table and load it:**



**```sql**

**CREATE TABLE papers (**

&#x20;   **id serial PRIMARY KEY,**

&#x20;   **arxiv\_id text,**

&#x20;   **title text,**

&#x20;   **abstract text,**

&#x20;   **categories text,**

&#x20;   **embedding vector(384)**

**);**

**```**



**Load with `rag\_pipeline/load\_to\_pgvector.py` (first 20k) and `rag\_pipeline/load\_extra.py` (remaining 80k). A paper's id is its row number across the two embedding files, 1 to 100000.**



**5. Create the size tiers:**



**```sql**

**CREATE TABLE papers\_1k   AS SELECT \* FROM papers WHERE id <= 1000;**

**CREATE TABLE papers\_10k  AS SELECT \* FROM papers WHERE id <= 10000;**

**CREATE TABLE papers\_20k  AS SELECT \* FROM papers WHERE id <= 20000;**

**CREATE TABLE papers\_50k  AS SELECT \* FROM papers WHERE id <= 50000;**

**CREATE TABLE papers\_100k AS SELECT \* FROM papers;**

**```**



**## Running**



**Run from the repo root. Every script asks for the Postgres password at runtime, and none stores it.**



**```**

**python dbms\_pgvector/export\_ground\_truth.py**

**python dbms\_pgvector/test\_backend.py**

**python dbms\_pgvector/exact\_baseline.py**

**python dbms\_pgvector/benchmark\_scale.py**

**python dbms\_pgvector/make\_charts.py**

**```**



**Note: `benchmark\_indexes.py` was run when the `papers` table held 20k rows. To reproduce it now, change its table name to `papers\_20k`.**



**## Shared interface**



**`pgvector\_backend.py` implements `common/backend\_interface.py` (`build\_index`, `search`, `count`) with L2 distance and the shared id scheme, so the same harness and `common/ground\_truth.json` can score pgvector and the dedicated vector database.**

