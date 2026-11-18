
import os
import json
import numpy as np

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery

# Configuration
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DATASET_PATH = r"C:\Users\Plantix\Downloads\arxiv-embeddings.json"
NAMESPACE = "arxiv"

NUM_QUERIES = 30
TOP_K = 5
DIMENSIONS = 384

if not API_KEY or not INDEX_NAME:
    raise ValueError("Pinecone configuration missing in .env")

# Connect to Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

print("Connected to Pinecone:", INDEX_NAME)

# Load dataset vectors and IDs
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
                f"Invalid dimensions for {paper['id']}"
            )

        paper_ids.append(str(paper["id"]))
        vectors.append(vector)

# Convert to NumPy matrix
matrix = np.vstack(vectors)

print("Dataset records:", len(paper_ids))
print("Vector dimensions:", matrix.shape[1])

# Normalize vectors for exact cosine similarity
norms = np.linalg.norm(matrix, axis=1)

if np.any(norms == 0):
    raise ValueError("Zero-length embedding found")

normalized_matrix = matrix / norms[:, None]

# Select 30 distinct queries
query_vectors = matrix[:NUM_QUERIES]
query_ids = paper_ids[:NUM_QUERIES]

recall_scores = []
all_results = []

print("\nStarting Recall@5 benchmark...\n")

for number, (query_id, query) in enumerate(
    zip(query_ids, query_vectors),
    start=1
):

    # Pinecone approximate search
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

    pinecone_ids = [
        match.id for match in response.matches
    ]

    # Exact cosine similarity against all 20,000 vectors
    query_norm = np.linalg.norm(query)

    similarities = (
        normalized_matrix @ (query / query_norm)
    )

    exact_indices = np.argsort(
        -similarities,
        kind="stable"
    )[:TOP_K]

    exact_ids = [
        paper_ids[i] for i in exact_indices
    ]

    # Calculate Recall@5
    matching_ids = set(pinecone_ids) & set(exact_ids)

    recall = len(matching_ids) / TOP_K

    recall_scores.append(recall)

    all_results.append({
        "query_id": query_id,
        "pinecone_ids": pinecone_ids,
        "exact_ids": exact_ids,
        "matching_ids": sorted(matching_ids),
        "recall_at_5": recall
    })

    print(
        f"Query {number}: "
        f"Recall@5 = {recall * 100:.1f}% "
        f"({len(matching_ids)}/{TOP_K})"
    )

# Overall results
average_recall = np.mean(recall_scores) * 100

print("\n========== PINECONE RECALL BENCHMARK ==========")
print("Dataset size:", len(paper_ids))
print("Dimensions:", DIMENSIONS)
print("Queries tested:", len(recall_scores))
print("Top K:", TOP_K)
print(f"Average Recall@5: {average_recall:.2f}%")

# Save detailed results
with open(
    "pinecone_recall_results.json",
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(
        all_results,
        output_file,
        indent=2
    )

print("\nDetailed results saved to pinecone_recall_results.json")
print("Recall benchmark completed!")
