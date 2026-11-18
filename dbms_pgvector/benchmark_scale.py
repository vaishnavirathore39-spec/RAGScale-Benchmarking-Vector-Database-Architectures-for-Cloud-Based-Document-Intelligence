import csv
import getpass
import math
import os
import statistics
import time

import psycopg2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(SCRIPT_DIR, "scale_results_median.csv")

TIERS = [("papers_1k", 1000), ("papers_10k", 10000), ("papers_20k", 20000),
         ("papers_50k", 50000), ("papers_100k", 100000)]
N_QUERIES = 50
K = 10
N_PASSES = 3

conn = psycopg2.connect(
    dbname="postgres", user="postgres",
    password=getpass.getpass("Postgres password: "),
    host="localhost", port="5432",
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SET maintenance_work_mem = '1GB'")   # same for every index build

# Fixed query set from the smallest tier, so every query exists in every tier
cur.execute("SELECT setseed(0.42)")
cur.execute("SELECT id, embedding::text FROM papers_1k ORDER BY random() LIMIT %s", (N_QUERIES,))
queries = cur.fetchall()


def run_queries(table, qs):
    """Return per-query latencies (ms) and result-id sets (query's own id removed)."""
    sql = f"SELECT id FROM {table} ORDER BY embedding <-> %s::vector LIMIT %s"
    latencies, results = [], []
    for qid, qvec in qs:
        start = time.perf_counter()
        cur.execute(sql, (qvec, K + 1))
        ids = [r[0] for r in cur.fetchall()]
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(set([i for i in ids if i != qid][:K]))
    return latencies, results


def timed(table, qs):
    """Full warm-up pass, then N_PASSES timed passes; latency = median per query."""
    run_queries(table, qs)                                   # warm-up (not counted)
    passes = [run_queries(table, qs) for _ in range(N_PASSES)]
    lat = [statistics.median(p[0][i] for p in passes) for i in range(len(qs))]
    return lat, passes[0][1]


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * len(values)))]


rows = []


def record(tier, config, setting, build_s, lat, recall):
    row = (tier, config, setting, round(build_s, 1),
           round(sum(lat) / len(lat), 2), round(p95(lat), 2), round(recall, 3))
    rows.append(row)
    print(f"{row[0]:<12}{row[1]:<22}{row[2]:<16}{row[3]:>9}{row[4]:>10}{row[5]:>10}{row[6]:>10}")


print(f"{'Tier':<12}{'Config':<22}{'Setting':<16}{'Build(s)':>9}{'Avg(ms)':>10}{'P95(ms)':>10}{'Recall':>10}")

for table, n in TIERS:
    cur.execute("DROP INDEX IF EXISTS bench_idx")

    # Exact search = ground truth and baseline
    exact_lat, truth = timed(table, queries)
    record(table, "No index (exact)", "-", 0.0, exact_lat, 1.0)

    lists = max(10, int(math.sqrt(n)))
    configs = [
        ("HNSW m=16 ef_c=64",
         f"CREATE INDEX bench_idx ON {table} USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64)",
         [("ef_search=40", "SET hnsw.ef_search = 40"),
          ("ef_search=100", "SET hnsw.ef_search = 100")]),
        (f"IVFFlat lists={lists}",
         f"CREATE INDEX bench_idx ON {table} USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists})",
         [("probes=10", "SET ivfflat.probes = 10"),
          ("probes=30", "SET ivfflat.probes = 30")]),
    ]

    for name, create_sql, settings in configs:
        start = time.time()
        cur.execute(create_sql)
        build_s = time.time() - start
        cur.execute("SET enable_seqscan = off")
        for label, set_sql in settings:
            cur.execute(set_sql)
            lat, res = timed(table, queries)
            recall = sum(len(r & t) / K for r, t in zip(res, truth)) / len(queries)
            record(table, name, label, build_s, lat, recall)
        cur.execute("SET enable_seqscan = on")
        cur.execute("DROP INDEX bench_idx")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["tier", "config", "setting", "build_s", "avg_ms", "p95_ms", f"recall_at_{K}"])
    w.writerows(rows)

print(f"\nSaved {OUT_CSV}")
conn.close()