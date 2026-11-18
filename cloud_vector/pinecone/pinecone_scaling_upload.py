
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone


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

TOTAL_RECORDS = 1000
BATCH_SIZE = 25
EXPECTED_DIMENSION = 384


if not API_KEY or not INDEX_NAME:
    raise ValueError("Pinecone credentials missing in .env")

if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset file not found")


# ==========================================
# 2. CONNECT TO PINECONE
# ==========================================

pc = Pinecone(api_key=API_KEY)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone:", INDEX_NAME)
print("Target namespace:", NAMESPACE)


# ==========================================
# 3. UPLOAD FIRST 1000 RECORDS
# ==========================================

batch = []

total_uploaded = 0
batch_number = 0

print("\nStarting 1K vector upload...\n")


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(file, start=1):

        if not line.strip():
            continue

        data = json.loads(line)

        embedding = [
            float(x) for x in data["embedding"]
        ]

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Invalid dimensions for {data['id']}"
            )

        document = {
           "_id": str(data["id"]),
            "embedding": embedding,
            "chunk_text": data.get("abstract") or "",
            "title": data.get("title") or "",
            "authors": data.get("authors") or "",
            "categories": data.get("categories") or "",
            "update_date": data.get("update_date") or ""
        }

        batch.append(document)

        if len(batch) == BATCH_SIZE:

            response = index.documents.upsert(
                namespace=NAMESPACE,
                documents=batch
            )

            uploaded = response.upserted_count

            total_uploaded += uploaded
            batch_number += 1

            print(
                f"Batch {batch_number}: "
                f"{uploaded} records uploaded"
            )

            batch = []

        if total_uploaded + len(batch) >= TOTAL_RECORDS:
            break


# Upload remaining records
if batch:

    response = index.documents.upsert(
        namespace=NAMESPACE,
        documents=batch
    )

    uploaded = response.upserted_count

    total_uploaded += uploaded
    batch_number += 1

    print(
        f"Batch {batch_number}: "
        f"{uploaded} records uploaded"
    )


# ==========================================
# 4. VERIFY UPLOAD
# ==========================================

print("\n========== UPLOAD SUMMARY ==========")

print("Target records:", TOTAL_RECORDS)
print("Successfully uploaded:", total_uploaded)
print("Total batches:", batch_number)
print("Namespace:", NAMESPACE)

stats = index.describe_index_stats()

print("\nPinecone index statistics:")
print(stats)

print("\n1K scaling upload completed!")
