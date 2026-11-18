import csv
import getpass
import json
import math
import os
import statistics
import time

import psycopg2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_FILE = os.path.join(SCRIPT_DIR, "..", "common", "ground_truth.json")
OUT_CSV = os.environ.get("OUT_CSV", os.path.join(SCRIPT_DIR, "scale_results_aws.csv"))

ALL_TIERS = [("papers_1k", 1000), ("papers_10k", 10000), ("papers_20k", 20000),
             ("papers_50k", 50000), ("papers_100k", 100000)]
wanted = os.environ.get("TIERS")              # e.g. TIERS=papers_1k,papers_10k
TIERS = [t for t in ALL_TIERS if not wanted or t[0] in wanted.split(",")]

K = 10
N_PASSES = 3
MAINT_MEM = os.environ.get("MAINT_MEM", "128MB")   # keep well below the instance RAM


def to_text(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"


conn = psycopg2.connect(
    dbname=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD") or getpass.getpass("Postgres password: "),
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"SET maintenance_work_mem = '{MAINT_MEM}'")
cur.execute("SET max_parallel_workers_per_gather = 0")   # single-process exact scans at every size

# Same 50 queries everywhere: taken from the shared ground-truth file
with open(GT_FILE, encoding="utf-8") as f:
    gt = json.load(f)
queries = [(q["id"], to_text(q["vector"])) for q in gt["queries"]]

cur.execute("SELECT version()")
print(cur.fetchone()[0])
cur.execute("SHOW shared_buffers")
print("shared_buffers:", cur.fetchone()[0], "| maintenance_work_mem:", MAINT_MEM)
cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
print("pgvector:", cur.fetchone()[0], "| queries:", len(queries))

new_file = not os.path.exists(OUT_CSV)
out = open(OUT_CSV, "a", newline="", encoding="utf-8")
writer = csv.writer(out)
if new_file:
    writer.writerow(["tier", "config", "setting", "build_s", "avg_ms", "p95_ms", f"recall_at_{K}"])
    out.flush()


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


def record(tier, config, setting, build_s, lat, recall):
    row = (tier, config, setting, round(build_s, 1),
           round(sum(lat) / len(lat), 2), round(p95(lat), 2), round(recall, 3))
    writer.writerow(row)
    out.flush()
    print(f"{row[0]:<12}{row[1]:<22}{row[2]:<16}{row[3]:>9}{row[4]:>10}{row[5]:>10}{row[6]:>10}")


print(f"\n{'Tier':<12}{'Config':<22}{'Setting':<16}{'Build(s)':>9}{'Avg(ms)':>10}{'P95(ms)':>10}{'Recall':>10}")

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
        try:
            start = time.time()
            cur.execute(create_sql)
            build_s = time.time() - start
            cur.execute("SET enable_seqscan = off")
            for label, set_sql in settings:
                cur.execute(set_sql)
                lat, res = timed(table, queries)
                recall = sum(len(r & t) / K for r, t in zip(res, truth)) / len(queries)
                record(table, name, label, build_s, lat, recall)
        except psycopg2.Error as e:
            print(f"FAILED {table} {name}: {str(e).strip()}")
            if conn.closed:
                raise SystemExit("Connection lost. Re-run with TIERS=<remaining tiers>.")
        finally:
            if not conn.closed:
                cur.execute("SET enable_seqscan = on")
                cur.execute("DROP INDEX IF EXISTS bench_idx")

out.close()
conn.close()
print(f"\nSaved {OUT_CSV}")