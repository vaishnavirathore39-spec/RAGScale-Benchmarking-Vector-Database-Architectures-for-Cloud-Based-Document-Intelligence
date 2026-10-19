import json
import os

from pgvector_backend import PgvectorBackend

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "..", "common", "ground_truth.json"), encoding="utf-8") as f:
    gt = json.load(f)
K = gt["k"]
TABLE = "papers_10k"

backend = PgvectorBackend(table=TABLE)
print("rows:", backend.count())
print("build seconds:", round(backend.build_index({"type": "hnsw", "m": 16, "ef_construction": 64}), 1))

total = 0.0
for q in gt["queries"]:
    res = backend.search(q["vector"], K + 1, {"ef_search": 40})
    ids = [r["id"] for r in res if r["id"] != q["id"]][:K]
    total += len(set(ids) & set(gt["truth"][TABLE][str(q["id"])])) / K
print("recall@10:", round(total / len(gt["queries"]), 3))

backend.build_index({"type": "none"})   # drop the test index
backend.close()