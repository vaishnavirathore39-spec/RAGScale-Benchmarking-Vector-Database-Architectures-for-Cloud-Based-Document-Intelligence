import getpass
import json
import os

import psycopg2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, "..", "common", "ground_truth.json")

TIERS = ["papers_1k", "papers_10k", "papers_20k", "papers_50k", "papers_100k"]
N_QUERIES = 50
K = 10


def to_text(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"


conn = psycopg2.connect(
    dbname="postgres", user="postgres",
    password=getpass.getpass("Postgres password: "),
    host="localhost", port="5432",
)
conn.autocommit = True
cur = conn.cursor()

# Same 50 queries as benchmark_scale.py
cur.execute("SELECT setseed(0.42)")
cur.execute("SELECT id, embedding::text FROM papers_1k ORDER BY random() LIMIT %s", (N_QUERIES,))
queries = [(qid, json.loads(vec)) for qid, vec in cur.fetchall()]

truth = {}
for table in TIERS:
    print(f"Exact search on {table}...")
    truth[table] = {}
    for qid, qvec in queries:
        cur.execute(
            f"SELECT id FROM {table} ORDER BY embedding <-> %s::vector LIMIT %s",
            (to_text(qvec), K + 1),
        )
        ids = [r[0] for r in cur.fetchall() if r[0] != qid][:K]
        truth[table][str(qid)] = ids

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "metric": "l2", "k": K, "seed": 0.42,
        "note": "queries are papers from papers_1k; truth excludes the query's own id",
        "queries": [{"id": qid, "vector": qvec} for qid, qvec in queries],
        "truth": truth,
    }, f)
print("Saved", OUT_FILE)
conn.close()