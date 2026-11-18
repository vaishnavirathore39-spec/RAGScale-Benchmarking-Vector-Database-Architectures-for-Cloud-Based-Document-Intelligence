
import os
import json
import time

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

NAMESPACE = "scale-10k"

TOTAL_RECORDS = 10000
BATCH_SIZE = 25
EXPECTED_DIMENSION = 384

MAX_RETRIES = 3


if not API_KEY or not INDEX_NAME:
    raise ValueError("Pinecone configuration missing in .env")

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
# 3. VERIFY EMPTY NAMESPACE
# ==========================================

stats = index.describe_index_stats()

namespace_stats = stats.namespaces.get(NAMESPACE)

if namespace_stats is None:
    raise ValueError("Namespace does not exist")

if namespace_stats.vector_count != 0:
    raise RuntimeError(
        f"Namespace already contains "
        f"{namespace_stats.vector_count} vectors. "
        "Stopping to avoid unintended re-upload."
    )

print("Namespace verified: empty")


# ==========================================
# 4. UPLOAD HELPER WITH RETRIES
# ==========================================

def upload_batch(batch, batch_number):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = index.documents.upsert(
                namespace=NAMESPACE,
                documents=batch
            )

            uploaded = response.upserted_count

            if uploaded != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)}, "
                    f"but Pinecone reported {uploaded}"
                )

            return uploaded

        except Exception:

            if attempt == MAX_RETRIES:
                raise

            print(
                f"Batch {batch_number} failed. "
                f"Retry {attempt}/{MAX_RETRIES}..."
            )

            time.sleep(2 * attempt)


# ==========================================
# 5. READ AND UPLOAD 10K RECORDS
# ==========================================

batch = []

total_uploaded = 0
batch_number = 0

print("\nStarting 10K vector upload...\n")


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        if not line.strip():
            continue

        if total_uploaded + len(batch) >= TOTAL_RECORDS:
            break

        data = json.loads(line)

        paper_id = str(data["id"])

        embedding = [
            float(x) for x in data["embedding"]
        ]

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Invalid dimensions for {paper_id}"
            )

        document = {
            "_id": paper_id,
            "embedding": embedding,
            "chunk_text": data.get("abstract") or "",
            "title": data.get("title") or "",
            "authors": data.get("authors") or "",
            "categories": data.get("categories") or "",
            "update_date": data.get("update_date") or ""
        }

        batch.append(document)

        if len(batch) == BATCH_SIZE:

            next_batch = batch_number + 1

            uploaded = upload_batch(
                batch,
                next_batch
            )

            total_uploaded += uploaded
            batch_number += 1

            print(
                f"Batch {batch_number}: "
                f"{uploaded} records uploaded | "
                f"Total: {total_uploaded}/{TOTAL_RECORDS}"
            )

            batch = []


# Upload remaining records
if batch:

    uploaded = upload_batch(
        batch,
        batch_number + 1
    )

    total_uploaded += uploaded
    batch_number += 1

    print(
        f"Batch {batch_number}: "
        f"{uploaded} records uploaded | "
        f"Total: {total_uploaded}/{TOTAL_RECORDS}"
    )


# ==========================================
# 6. VERIFY UPLOAD
# ==========================================

print("\nChecking Pinecone statistics...")

final_stats = index.describe_index_stats()

final_namespace = final_stats.namespaces.get(NAMESPACE)

final_count = (
    final_namespace.vector_count
    if final_namespace else 0
)


# ==========================================
# 7. DISPLAY SUMMARY
# ==========================================

print("\n========== 10K UPLOAD SUMMARY ==========")

print("Target records:", TOTAL_RECORDS)
print("Successfully submitted:", total_uploaded)
print("Total batches:", batch_number)
print("Namespace:", NAMESPACE)
print("Verified namespace count:", final_count)

print(
    "Total index vectors:",
    final_stats.total_vector_count
)

if total_uploaded == TOTAL_RECORDS and final_count == TOTAL_RECORDS:
    print("\n10K scaling upload completed successfully!")

else:
    print("\nWARNING: Upload count does not match target.")
    print("Check Pinecone namespace statistics.")


# ==========================================
# 8. SAVE UPLOAD SUMMARY
# ==========================================

summary = {
    "database": "Pinecone",
    "index": INDEX_NAME,
    "namespace": NAMESPACE,
    "target_records": TOTAL_RECORDS,
    "submitted_records": total_uploaded,
    "verified_records": final_count,
    "batch_size": BATCH_SIZE,
    "total_batches": batch_number,
    "dimensions": EXPECTED_DIMENSION
}

output_path = (
    BASE_DIR / "pinecone_scaling_10k_upload_summary.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as output_file:

    json.dump(summary, output_file, indent=2)

print("Summary saved:", output_path)
