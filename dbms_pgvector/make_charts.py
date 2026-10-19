import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "scale_results_median.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "charts")
os.makedirs(OUT_DIR, exist_ok=True)

ROWS = {"papers_1k": 1000, "papers_10k": 10000, "papers_20k": 20000,
        "papers_50k": 50000, "papers_100k": 100000}
# Exact scan, single process (exact_baseline.py), so every size is measured the same way
EXACT_SINGLE = {1000: 5.55, 10000: 181.55, 20000: 227.54, 50000: 672.77, 100000: 2390.62}

latency, recall, build = {}, {}, {}
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        n = ROWS[r["tier"]]
        if r["config"].startswith("No index"):
            continue
        fam = "HNSW" if r["config"].startswith("HNSW") else "IVFFlat"
        label = f"{fam} ({r['setting']})"
        latency.setdefault(label, {})[n] = float(r["avg_ms"])
        recall.setdefault(label, {})[n] = float(r["recall_at_10"])
        build.setdefault(fam, {})[n] = float(r["build_s"])
latency["Exact scan (single process)"] = EXACT_SINGLE


def plot(series, ylabel, title, fname, logy=False):
    plt.figure(figsize=(7, 4.5))
    for label, pts in series.items():
        xs = sorted(pts)
        plt.plot(xs, [pts[x] for x in xs], marker="o", label=label)
    plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel("Rows in table")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


plot(latency, "Average query latency (ms)", "Latency vs data size", "latency.png", logy=True)
plot(recall, "Recall@10", "Recall vs data size", "recall.png")
plot(build, "Index build time (s)", "Build time vs data size", "build_time.png")
print("Saved charts to", OUT_DIR)