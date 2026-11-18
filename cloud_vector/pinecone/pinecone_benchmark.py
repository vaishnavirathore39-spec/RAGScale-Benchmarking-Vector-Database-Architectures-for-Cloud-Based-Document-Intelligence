
import os
import json
import time
import math
import statistics

from datetime import datetime, timezone
from pathlib import Path

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

NAMESPACE = "arxiv"

TOP_K = 5
NUM_QUERIES = 30
EXPECTED_DIMENSION = 384


if not API_KEY or not INDEX_NAME:
    raise ValueError(
        "Pinecone configuration missing in .env"
    )

if not DATASET_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


# ==========================================
# 2. CONNECT TO PINECONE
# ==========================================

pc = Pinecone(api_key=API_KEY)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone:", INDEX_NAME)


# ==========================================
# 3. LOAD QUERY VECTORS
# ==========================================

query_vectors = []

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        if not line.strip():
            continue

        data = json.loads(line)

        embedding = [
            float(x) for x in data["embedding"]
        ]

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Invalid dimensions for ID {data['id']}"
            )

        query_vectors.append({
            "id": data["id"],
            "title": data["title"],
            "embedding": embedding
        })

        if len(query_vectors) == NUM_QUERIES:
            break


if len(query_vectors) < NUM_QUERIES:
    raise ValueError(
        "Not enough query records in dataset"
    )

print("Queries loaded:", len(query_vectors))


# ==========================================
# 4. WARM-UP QUERY
# ==========================================

warmup = query_vectors[0]

index.documents.search(
    namespace=NAMESPACE,
    top_k=TOP_K,
    score_by=[
        DenseVectorQuery(
            field="embedding",
            values=warmup["embedding"]
        )
    ],
    include_fields=["title"]
)

print("Warm-up completed.")


# ==========================================
# 5. RUN BENCHMARK
# ==========================================

latencies = []

successful_queries = 0
failed_queries = 0

print("\nStarting Pinecone benchmark...\n")


for number, query in enumerate(
    query_vectors,
    start=1
):

    start = time.perf_counter()

    try:

        results = index.documents.search(
            namespace=NAMESPACE,
            top_k=TOP_K,
            score_by=[
                DenseVectorQuery(
                    field="embedding",
                    values=query["embedding"]
                )
            ],
            include_fields=["title"]
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

        successful_queries += 1

        print(
            f"Query {number}: "
            f"{elapsed:.3f} ms | "
            f"Results: {len(results.matches)}"
        )

    except Exception as error:

        failed_queries += 1

        print(
            f"Query {number} failed: {error}"
        )


if not latencies:
    raise RuntimeError(
        "No benchmark queries succeeded"
    )


# ==========================================
# 6. CALCULATE STATISTICS
# ==========================================

sorted_latencies = sorted(latencies)

average_latency = statistics.mean(latencies)

p50 = statistics.median(sorted_latencies)

# Nearest-rank P95
p95_position = (
    math.ceil(0.95 * len(sorted_latencies)) - 1
)

p95 = sorted_latencies[p95_position]

minimum_latency = min(latencies)

maximum_latency = max(latencies)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== PINECONE BENCHMARK ==========")

print("Index:", INDEX_NAME)
print("Namespace:", NAMESPACE)
print("Dataset size: 20000")
print("Vector dimensions:", EXPECTED_DIMENSION)
print("Top K:", TOP_K)

print("Successful queries:", successful_queries)
print("Failed queries:", failed_queries)

print(
    f"Average latency: {average_latency:.3f} ms"
)

print(
    f"Median latency (P50): {p50:.3f} ms"
)

print(
    f"P95 latency: {p95:.3f} ms"
)

print(
    f"Minimum latency: {minimum_latency:.3f} ms"
)

print(
    f"Maximum latency: {maximum_latency:.3f} ms"
)


# ==========================================
# 8. PREPARE BENCHMARK DATA
# ==========================================

benchmark_data = {

    "database": "Pinecone",

    "index": INDEX_NAME,

    "namespace": NAMESPACE,

    "dataset_size": 20000,

    "dimensions": EXPECTED_DIMENSION,

    "top_k": TOP_K,

    "num_queries": NUM_QUERIES,

    "successful_queries": successful_queries,

    "failed_queries": failed_queries,

    "average_latency_ms": average_latency,

    "median_latency_ms": p50,

    "p95_latency_ms": p95,

    "minimum_latency_ms": minimum_latency,

    "maximum_latency_ms": maximum_latency,

    "individual_latencies_ms": latencies,

    "tested_at": datetime.now(
        timezone.utc
    ).isoformat()

}


# ==========================================
# 9. SAVE RESULTS IN PROJECT DIRECTORY
# ==========================================

output_path = (
    BASE_DIR / "pinecone_latency_results.json"
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


print("\nLatency results saved successfully!")

print("Saved location:", output_path)

print("\nBenchmark completed!")
