
import os
import json
import time

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery

# Load environment variables
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DATASET_PATH = r"C:\Users\Plantix\Downloads\arxiv-embeddings.json"
NAMESPACE = "arxiv"

if not API_KEY or not INDEX_NAME:
    raise ValueError("Pinecone configuration missing in .env")

# Connect to Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# Read one existing query embedding
with open(DATASET_PATH, "r", encoding="utf-8") as file:
    query_paper = json.loads(file.readline())

query_vector = [float(x) for x in query_paper["embedding"]]

print("Query paper:", query_paper["title"])
print("Query dimensions:", len(query_vector))

# Perform vector similarity search
start_time = time.perf_counter()

results = index.documents.search(
    namespace=NAMESPACE,
    top_k=5,
    score_by=[
        DenseVectorQuery(
            field="embedding",
            values=query_vector
        )
    ],
    include_fields=[
        "title",
        "authors",
        "categories"
    ]
)

end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000

# Display results
print("\n========== TOP 5 SIMILAR PAPERS ==========")

for rank, match in enumerate(results.matches, start=1):
    print(f"\nRank: {rank}")
    print("Paper ID:", match.id)
    print("Title:", match.get("title", "N/A"))
    print("Authors:", match.get("authors", "N/A"))
    print("Similarity Score:", match.score)

# Search summary
print("\n========== SEARCH SUMMARY ==========")
print("Index:", INDEX_NAME)
print("Namespace:", NAMESPACE)
print("Top K: 5")
print(f"Query latency: {latency_ms:.3f} ms")
print("Search completed successfully!")
