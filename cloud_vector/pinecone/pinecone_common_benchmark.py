
import json
import math
import statistics
import time
from pathlib import Path

from cloud_vector.pinecone.pinecone_backend import PineconeBackend

BASE_DIR = Path(__file__).resolve().parents[2]
GT_PATH = BASE_DIR / "common" / "ground_truth.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "pinecone_common_results.json"

with open(GT_PATH, encoding="utf-8") as f:
    gt = json.load(f)

queries = gt["queries"]
top_k = gt["k"]
truth = gt["truth"]["papers_20k"]

backend = PineconeBackend(namespace="arxiv")

latencies = []
recalls = []

try:
    print("========== PINECONE COMMON BENCHMARK ==========")
    print("Namespace:", backend.namespace)
    print("Dataset size:", backend.count())
    print("Queries:", len(queries))
    print("Top K:", top_k)

    # One warm-up query
    backend.search(queries[0]["vector"], top_k + 1)
    print("Warm-up completed.\n")

    for number, query in enumerate(queries, start=1):
        query_id = int(query["id"])

        start = time.perf_counter()

        results = backend.search(
            query["vector"],
            top_k + 1
        )

        latency = (time.perf_counter() - start) * 1000

        # Exclude the query's own document
        results = [
            r for r in results
            if r["id"] != query_id
        ][:top_k]

        actual_ids = {r["id"] for r in results}
        expected_ids = set(truth[str(query_id)])

        recall = len(actual_ids & expected_ids) / top_k

        latencies.append(latency)
        recalls.append(recall)

        print(
            f"Query {number}: "
            f"Latency={latency:.3f} ms | "
            f"Recall@{top_k}={recall * 100:.1f}%"
        )

    ordered = sorted(latencies)

    average = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    average_recall = statistics.mean(recalls) * 100

    summary = {
        "backend": "Pinecone",
        "namespace": backend.namespace,
        "dataset_size": backend.count(),
        "queries": len(queries),
        "top_k": top_k,
        "average_latency_ms": average,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "average_recall_percent": average_recall,
        "individual_results": [
            {
                "query_id": int(q["id"]),
                "latency_ms": latency,
                "recall": recall
            }
            for q, latency, recall
            in zip(queries, latencies, recalls)
        ]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Average latency: {average:.3f} ms")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print(f"Average Recall@{top_k}: {average_recall:.2f}%")
    print("Results saved:", OUTPUT_PATH)

finally:
    backend.close()
