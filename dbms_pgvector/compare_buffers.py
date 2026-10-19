import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load(name):
    out = {}
    with open(os.path.join(SCRIPT_DIR, name), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["tier"], r["config"], r["setting"])] = r
    return out


a = load("scale_results_median.csv")   # shared_buffers = 128 MB
b = load("scale_results_1gb.csv")      # shared_buffers = 1 GB

print(f"{'Tier':<12}{'Config':<22}{'Setting':<16}{'128MB ms':>10}{'1GB ms':>10}{'Change':>9}")
for key in a:
    if key in b and key[0] in ("papers_50k", "papers_100k"):
        x, y = float(a[key]["avg_ms"]), float(b[key]["avg_ms"])
        print(f"{key[0]:<12}{key[1]:<22}{key[2]:<16}{x:>10.2f}{y:>10.2f}{(y - x) / x * 100:>8.0f}%")