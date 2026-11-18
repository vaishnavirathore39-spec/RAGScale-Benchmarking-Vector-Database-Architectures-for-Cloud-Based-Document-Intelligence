
import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Load configuration
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

DATASET_PATH = r"C:\Users\Plantix\Downloads\arxiv-embeddings.json"

NAMESPACE = "arxiv"
BATCH_SIZE = 100
EXPECTED_DIMENSION = 384

if not API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")

if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME missing in .env")

# Connect to Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

print("Connected to Pinecone!")
print("Index:", INDEX_NAME)

def upload_batch(batch, batch_number):
    response = index.documents.upsert(
        namespace=NAMESPACE,
        documents=batch
    )

    uploaded = response.upserted_count

    print(
        f"Batch {batch_number}: "
        f"{uploaded} records uploaded"
    )

    return uploaded


def main():

    batch = []
    batch_number = 0
    total_uploaded = 0
    total_records = 0

    print("\nStarting arXiv embedding upload...\n")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            data = json.loads(line)

            paper_id = str(data["id"])
            embedding = data["embedding"]

            if len(embedding) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Wrong dimensions for ID {paper_id}"
                )

            # Build Pinecone document
            document = {
                "_id": paper_id,
                "embedding": [float(x) for x in embedding],
                "chunk_text": data.get("abstract") or "",
                "title": data.get("title") or "",
                "authors": data.get("authors") or "",
                "categories": data.get("categories") or "",
                "update_date": data.get("update_date") or ""
            }

            batch.append(document)
            total_records += 1

            if len(batch) >= BATCH_SIZE:

                batch_number += 1

                total_uploaded += upload_batch(
                    batch,
                    batch_number
                )

                batch = []

        # Upload remaining records
        if batch:

            batch_number += 1

            total_uploaded += upload_batch(
                batch,
                batch_number
            )

    print("\n========== UPLOAD SUMMARY ==========")
    print("Total source records:", total_records)
    print("Successfully submitted:", total_uploaded)
    print("Total batches:", batch_number)
    print("Namespace:", NAMESPACE)
    print("Index:", INDEX_NAME)

    print("\nChecking Pinecone index statistics...")

    time.sleep(5)

    stats = index.describe_index_stats()

    print("Pinecone vector count:", stats.total_vector_count)
    print("Upload process completed!")


if __name__ == "__main__":
    main()
