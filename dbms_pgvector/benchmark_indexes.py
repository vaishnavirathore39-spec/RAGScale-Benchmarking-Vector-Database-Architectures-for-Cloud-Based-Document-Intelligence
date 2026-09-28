import time
import getpass
import psycopg2

conn = psycopg2.connect(
    dbname="postgres", user="postgres",
    password=getpass.getpass("Postgres password: "),
    host="localhost", port="5432",
)
conn.autocommit = True
cur = conn.cursor()

N_QUERIES = 50
K = 10

CONFIGS = [
    ("HNSW m=16 ef_c=64",
     "CREATE INDEX bench_idx ON papers USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64)",
     "SET hnsw.ef_search = 40"),
    ("HNSW m=24 ef_c=100",
     "CREATE INDEX bench_idx ON papers USING hnsw (embedding vector_l2_ops) WITH (m = 24, ef_construction = 100)",
     "SET hnsw.ef_search = 40"),
    ("IVFFlat lists=50",
     "CREATE INDEX bench_idx ON papers USING ivfflat (embedding vector_l2_ops) WITH (lists = 50)",
     "SET ivfflat.probes = 10"),
    ("IVFFlat lists=100",
     "CREATE INDEX bench_idx ON papers USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)",
     "SET ivfflat.probes = 10"),
    ("IVFFlat lists=200",
     "CREATE INDEX bench_idx ON papers USING ivfflat (embedding vector_l2_ops) WITH (lists = 200)",
     "SET ivfflat.probes = 10"),
]

SEARCH_SQL = "SELECT id FROM papers ORDER BY embedding <-> %s::vector LIMIT %s"

def run_queries(queries):
    latencies, results = [], []
    for q in queries:
        start = time.perf_counter()
        cur.execute(SEARCH_SQL, (q, K))
        ids = {r[0] for r in cur.fetchall()}
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(ids)
    return latencies, results

def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * len(values)))]

# Start from a clean slate (removes idx_hnsw_default; recreated at the end)
cur.execute("DROP INDEX IF EXISTS idx_hnsw_default, bench_idx")

cur.execute("SELECT embedding::text FROM papers ORDER BY random() LIMIT %s", (N_QUERIES,))
queries = [row[0] for row in cur.fetchall()]

print("Computing exact ground truth (no index)...")
exact_lat, truth = run_queries(queries)
rows = [("No index (exact)", 0.0, sum(exact_lat) / len(exact_lat), p95(exact_lat), 1.0)]

for name, create_sql, setting in CONFIGS:
    print(f"Testing {name}...")
    start = time.time()
    cur.execute(create_sql)
    build_s = time.time() - start
    cur.execute(setting)
    cur.execute("SET enable_seqscan = off")
    run_queries(queries[:5])  # warm-up
    lat, res = run_queries(queries)
    recall = sum(len(r & t) / K for r, t in zip(res, truth)) / len(queries)
    rows.append((name, build_s, sum(lat) / len(lat), p95(lat), recall))
    cur.execute("SET enable_seqscan = on")
    cur.execute("DROP INDEX bench_idx")

print(f"\n{'Config':<22}{'Build (s)':>10}{'Avg (ms)':>10}{'P95 (ms)':>10}{'Recall@10':>11}")
for name, build, avg, p, rec in rows:
    print(f"{name:<22}{build:>10.1f}{avg:>10.2f}{p:>10.2f}{rec:>11.3f}")

cur.execute("CREATE INDEX idx_hnsw_default ON papers USING hnsw (embedding vector_l2_ops)")
print("\nRecreated idx_hnsw_default.")
conn.close()