
import csv
import json
from pathlib import Path

# Original dataset containing the first 20,000 papers
DATASET_PATH = Path(
    r"C:\Users\Plantix\Downloads\arxiv-embeddings.json"
)

OUTPUT_PATH = Path(__file__).resolve().parent / "id_mapping.csv"

EXPECTED_RECORDS = 20000

def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    seen_ids = set()
    count = 0

    with open(
        DATASET_PATH, "r", encoding="utf-8"
    ) as source, open(
        OUTPUT_PATH, "w", newline="", encoding="utf-8"
    ) as output:

        writer = csv.writer(output)

        writer.writerow([
            "numeric_id",
            "arxiv_id"
        ])

        for numeric_id, line in enumerate(source, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            arxiv_id = str(record["id"])

            if arxiv_id in seen_ids:
                raise ValueError(
                    f"Duplicate arXiv ID found: {arxiv_id}"
                )

            seen_ids.add(arxiv_id)

            writer.writerow([
                numeric_id,
                arxiv_id
            ])

            count += 1

    print("\n========== ID MAPPING SUMMARY ==========")
    print("Records mapped:", count)
    print("Expected records:", EXPECTED_RECORDS)
    print("Output file:", OUTPUT_PATH)

    if count != EXPECTED_RECORDS:
        raise ValueError(
            "Record count does not match expected dataset size."
        )

    print("\nID mapping generated successfully!")


if __name__ == "__main__":
    main()
