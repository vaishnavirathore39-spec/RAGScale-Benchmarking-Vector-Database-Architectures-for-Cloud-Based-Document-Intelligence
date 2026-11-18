
import os
import json
import time
import math
import statistics
import numpy as np

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DATASET_PATH = Path(
    r"C:\Users\Plantix\Downloads\arxiv-embeddings.json"
)

NAMESPACE = "scale-1k"

DATASET_SIZE = 1000
NUM_QUERIES = 30
TOP_K = 5
DIMENSIONS = 384

if not API_KEY or not INDEX_NAME:
    raise ValueError("Pinecone credentials missing")

if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset not found")


# ==========================================
# 2. CONNECT TO PINECONE
# ==========================================

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

print("Connected to Pinecone:", INDEX_NAME)
print("Namespace:", NAMESPACE)


# ==========================================
# 3. LOAD FIRST 1000 VECTORS
# ==========================================

paper_ids = []
vectors = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:

    for line in file:

        if not line.strip():
            continue

        paper = json.loads(line)

        vector = np.asarray(
            paper["embedding"],
            dtype=np.float32
        )

        if len(vector) != DIMENSIONS:
            raise ValueError(
                f"Invalid dimensions: {paper['id']}"
            )

        paper_ids.append(str(paper["id"]))
        vectors.append(vector)

        if len(paper_ids) == DATASET_SIZE:
            break


matrix = np.vstack(vectors)

if len(paper_ids) != DATASET_SIZE:
    raise ValueError("Expected 1000 dataset records")


print("Dataset records:", len(paper_ids))
print("Dimensions:", matrix.shape[1])


# ==========================================
# 4. VERIFY NAMESPACE COUNT
# ==========================================

stats = index.describe_index_stats()

namespace_stats = stats.namespaces.get(NAMESPACE)

if namespace_stats is None:
    raise ValueError("Scaling namespace not found")

if namespace_stats.vector_count != DATASET_SIZE:
    raise ValueError(
        f"Expected {DATASET_SIZE} vectors, "
        f"found {namespace_stats.vector_count}"
    )

print("Verified namespace vector count:", namespace_stats.vector_count)


# ==========================================
# 5. PREPARE EXACT SEARCH
# ==========================================

norms = np.linalg.norm(matrix, axis=1)

if np.any(norms == 0):
    raise ValueError("Zero-length embedding found")

normalized_matrix = matrix / norms[:, None]

query_vectors = matrix[:NUM_QUERIES]
query_ids = paper_ids[:NUM_QUERIES]


# ==========================================
# 6. WARM-UP QUERY
# ==========================================

warmup = query_vectors[0]

index.documents.search(
    namespace=NAMESPACE,
    top_k=TOP_K,
    score_by=[
        DenseVectorQuery(
            field="embedding",
            values=warmup.tolist()
        )
    ]
)

print("Warm-up completed.")


# ==========================================
# 7. RUN SCALING BENCHMARK
# ==========================================

latencies = []
recall_scores = []

query_results = []

print("\nStarting 1K scaling benchmark...\n")


for number, (query_id, query) in enumerate(
    zip(query_ids, query_vectors),
    start=1
):

    # Pinecone search latency
    start = time.perf_counter()

    response = index.documents.search(
        namespace=NAMESPACE,
        top_k=TOP_K,
        score_by=[
            DenseVectorQuery(
                field="embedding",
                values=query.tolist()
            )
        ]
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    pinecone_ids = [
        match.id for match in response.matches
    ]

    latencies.append(elapsed)

    # Exact cosine similarity over 1000 vectors
    similarities = (
        normalized_matrix @
        (query / np.linalg.norm(query))
    )

    exact_indices = np.argsort(
        -similarities,
        kind="stable"
    )[:TOP_K]

    exact_ids = [
        paper_ids[i] for i in exact_indices
    ]

    # Recall@5
    matching_ids = (
        set(pinecone_ids) & set(exact_ids)
    )

    recall = len(matching_ids) / TOP_K

    recall_scores.append(recall)

    query_results.append({
        "query_id": query_id,
        "latency_ms": elapsed,
        "recall_at_5": recall,
        "matching_results": len(matching_ids)
    })

    print(
        f"Query {number}: "
        f"Latency = {elapsed:.3f} ms | "
        f"Recall@5 = {recall * 100:.1f}%"
    )


# ==========================================
# 8. CALCULATE STATISTICS
# ==========================================

sorted_latencies = sorted(latencies)

average_latency = statistics.mean(latencies)
p50 = statistics.median(sorted_latencies)

p95_position = (
    math.ceil(0.95 * len(sorted_latencies)) - 1
)

p95 = sorted_latencies[p95_position]

average_recall = (
    statistics.mean(recall_scores) * 100
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========== 1K PINECONE BENCHMARK ==========")

print("Index:", INDEX_NAME)
print("Namespace:", NAMESPACE)
print("Dataset size:", DATASET_SIZE)
print("Dimensions:", DIMENSIONS)
print("Queries:", NUM_QUERIES)
print("Top K:", TOP_K)

print(f"Average latency: {average_latency:.3f} ms")
print(f"P50 latency: {p50:.3f} ms")
print(f"P95 latency: {p95:.3f} ms")
print(f"Minimum latency: {min(latencies):.3f} ms")
print(f"Maximum latency: {max(latencies):.3f} ms")

print(f"Average Recall@5: {average_recall:.2f}%")


# ==========================================
# 10. SAVE RESULTS
# ==========================================

benchmark_data = {

    "database": "Pinecone",

    "experiment": "1K scaling",

    "index": INDEX_NAME,

    "namespace": NAMESPACE,

    "dataset_size": DATASET_SIZE,

    "dimensions": DIMENSIONS,

    "num_queries": NUM_QUERIES,

    "top_k": TOP_K,

    "average_latency_ms": average_latency,

    "median_latency_ms": p50,

    "p95_latency_ms": p95,

    "minimum_latency_ms": min(latencies),

    "maximum_latency_ms": max(latencies),

    "average_recall_at_5_percent": average_recall,

    "individual_results": query_results,

    "tested_at": datetime.now(
        timezone.utc
    ).isoformat()

}


output_path = (
    BASE_DIR / "pinecone_scaling_1k_results.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        benchmark_data,
        output_file,
        indent=2
    )


print("\nResults saved successfully!")
print("Saved location:", output_path)
print("\n1K scaling benchmark completed!")
