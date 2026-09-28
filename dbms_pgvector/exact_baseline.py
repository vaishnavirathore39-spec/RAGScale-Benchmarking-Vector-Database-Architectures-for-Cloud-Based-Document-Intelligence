import getpass
import statistics
import time

import psycopg2

TIERS = ["papers_1k", "papers_10k", "papers_20k", "papers_50k", "papers_100k"]
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
cur.execute("SET max_parallel_workers_per_gather = 0")   # single-process scans for every tier

# Same query set as benchmark_scale.py
cur.execute("SELECT setseed(0.42)")
cur.execute("SELECT id, embedding::text FROM papers_1k ORDER BY random() LIMIT %s", (N_QUERIES,))
queries = cur.fetchall()


def run(table):
    sql = f"SELECT id FROM {table} ORDER BY embedding <-> %s::vector LIMIT %s"
    lat = []
    for qid, qvec in queries:
        start = time.perf_counter()
        cur.execute(sql, (qvec, K + 1))
        cur.fetchall()
        lat.append((time.perf_counter() - start) * 1000)
    return lat


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * len(values)))]


print(f"{'Tier':<14}{'Avg(ms)':>10}{'P95(ms)':>10}")
for t in TIERS:
    run(t)                                   # warm-up pass
    passes = [run(t) for _ in range(N_PASSES)]
    per_query = [statistics.median(p[i] for p in passes) for i in range(len(queries))]
    print(f"{t:<14}{sum(per_query) / len(per_query):>10.2f}{p95(per_query):>10.2f}")

conn.close()