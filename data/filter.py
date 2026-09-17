import json

# --- Settings you can tweak ---
CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]   # pick categories relevant to your project
MAX_RECORDS = 20000                          # target size (aim for 10k-100k)
INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "arxiv-filtered.json"

kept = 0
scanned = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        scanned += 1
        record = json.loads(line)

        # 'categories' field looks like "cs.AI cs.LG" (space-separated, can have multiple)
        record_categories = record.get("categories", "").split()

        if any(cat in CATEGORIES for cat in record_categories):
            outfile.write(json.dumps(record) + "\n")
            kept += 1

        if kept >= MAX_RECORDS:
            break

        if scanned % 100000 == 0:
            print(f"Scanned {scanned}, kept {kept} so far...")

print(f"\nDone. Scanned {scanned} records, kept {kept} in {OUTPUT_FILE}")