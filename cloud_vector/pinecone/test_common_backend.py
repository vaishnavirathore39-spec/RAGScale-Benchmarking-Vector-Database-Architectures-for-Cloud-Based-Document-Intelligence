
import json

from cloud_vector.pinecone.pinecone_backend import PineconeBackend


# Load shared ground truth
with open(
    "common/ground_truth.json",
    "r",
    encoding="utf-8"
) as file:
    ground_truth = json.load(file)


query = ground_truth["queries"][0]
query_id = int(query["id"])
query_vector = query["vector"]
top_k = ground_truth["k"]

backend = PineconeBackend(namespace="arxiv")

try:
    print("\n========== COMMON SEARCH TEST ==========")

    print("Backend:", backend.name)
    print("Query ID:", query_id)
    print("Top K:", top_k)

    results = backend.search(
        query_vector,
        top_k + 1
    )

    # Remove the query's own ID, matching the shared harness
    results = [
        result for result in results
        if result["id"] != query_id
    ][:top_k]

    expected = set(
        ground_truth["truth"]["papers_20k"][str(query_id)]
    )

    actual = {
        result["id"] for result in results
    }

    recall = len(actual & expected) / top_k

    print("\nRetrieved results:")

    for result in results:
        print(
            "ID:", result["id"],
            "| Distance:", round(result["distance"], 6)
        )

    print("\nRecall@10:", round(recall, 3))

    print("\nCommon search test completed!")

finally:
    backend.close()
