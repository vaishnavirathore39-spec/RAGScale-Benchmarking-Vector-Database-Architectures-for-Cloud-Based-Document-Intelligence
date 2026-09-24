import json
import os
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "arxiv-filtered.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "arxiv-embeddings.json")

model = SentenceTransformer('all-MiniLM-L6-v2')

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} records. Generating embeddings...")

texts = [r["title"] + " " + r["abstract"] for r in records]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding.tolist()
        out.write(json.dumps(record) + "\n")

print(f"Done. Wrote {len(records)} embeddings to {OUTPUT_FILE}")