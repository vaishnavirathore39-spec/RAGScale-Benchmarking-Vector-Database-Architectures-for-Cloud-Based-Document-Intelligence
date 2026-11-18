
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from rag_pipeline.s3_document_loader import load_documents


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "all-MiniLM-L6-v2"

USE_S3 = False  # Keep False for local testing

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "sample_embeddings.jsonl"
)


# ==========================================
# LOAD DOCUMENTS
# ==========================================

print("Loading documents...")

documents = load_documents(use_s3=USE_S3)

if not documents:
    raise ValueError("No documents found!")

print("Documents loaded:", len(documents))


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

texts = [
    document["text"]
    for document in documents
]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding generation completed!")


# ==========================================
# SAVE EMBEDDINGS
# ==========================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as output:

    for document, embedding in zip(documents, embeddings):

        record = {
            "source": document["source"],
            "text": document["text"],
            "embedding": embedding.tolist(),
            "dimensions": len(embedding)
        }

        output.write(json.dumps(record) + "\n")


# ==========================================
# SUMMARY
# ==========================================

print("\n========== INGESTION SUMMARY ==========")

print("Documents processed:", len(documents))
print("Embedding model:", MODEL_NAME)
print("Embedding dimensions:", embeddings.shape[1])
print("Output file:", OUTPUT_FILE)

print("\nLocal ingestion completed successfully!")
