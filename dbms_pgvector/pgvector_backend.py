import getpass
import os
import sys
import time

import psycopg2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from backend_interface import VectorBackend


def to_text(vec):
    return "[" + ",".join(str(x) for x in vec) + "]"


class PgvectorBackend(VectorBackend):
    name = "pgvector"
    INDEX = "backend_idx"

    def __init__(self, table="papers_100k", host="localhost", dbname="postgres",
                 user="postgres", password=None, port="5432"):
        self.table = table
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, host=host, port=port,
            password=password or getpass.getpass("Postgres password: "),
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.cur.execute("SET maintenance_work_mem = '1GB'")
        self._applied = {}

    def build_index(self, config):
        self.cur.execute(f"DROP INDEX IF EXISTS {self.INDEX}")
        self._applied = {}
        kind = config["type"]
        start = time.time()
        if kind == "hnsw":
            m = int(config.get("m", 16))
            efc = int(config.get("ef_construction", 64))
            self.cur.execute(
                f"CREATE INDEX {self.INDEX} ON {self.table} USING hnsw "
                f"(embedding vector_l2_ops) WITH (m = {m}, ef_construction = {efc})")
        elif kind == "ivfflat":
            lists = int(config["lists"])
            self.cur.execute(
                f"CREATE INDEX {self.INDEX} ON {self.table} USING ivfflat "
                f"(embedding vector_l2_ops) WITH (lists = {lists})")
        elif kind != "none":
            raise ValueError(f"unknown index type: {kind}")
        build_s = time.time() - start
        # Use the index whenever one exists, like benchmark_scale.py does
        self.cur.execute("SET enable_seqscan = " + ("off" if kind != "none" else "on"))
        return build_s

    def _apply(self, params):
        for key, setting in (("ef_search", "hnsw.ef_search"), ("probes", "ivfflat.probes")):
            if key in params and self._applied.get(key) != params[key]:
                self.cur.execute(f"SET {setting} = {int(params[key])}")
                self._applied[key] = params[key]

    def search(self, vector, top_k, params=None):
        self._apply(params or {})
        q = to_text(vector)
        self.cur.execute(
            f"SELECT id, embedding <-> %s::vector FROM {self.table} "
            f"ORDER BY embedding <-> %s::vector LIMIT %s", (q, q, int(top_k)))
        return [{"id": r[0], "distance": float(r[1])} for r in self.cur.fetchall()]

    def count(self):
        self.cur.execute(f"SELECT count(*) FROM {self.table}")
        return self.cur.fetchone()[0]

    def close(self):
        self.conn.close()