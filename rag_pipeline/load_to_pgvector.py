import json
import psycopg2

DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Rit123&&",   # replace with your actual postgres password
    "host": "localhost",
    "port": "5432"
}

INPUT_FILE = "arxiv-embeddings.json"

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

count = 0
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        arxiv_id = record.get("id", "")
        title = record.get("title", "")
        abstract = record.get("abstract", "")
        categories = record.get("categories", "")
        embedding = record.get("embedding", [])

        cur.execute(
            """
            INSERT INTO papers (arxiv_id, title, abstract, categories, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (arxiv_id, title, abstract, categories, embedding)
        )

        count += 1
        if count % 1000 == 0:
            conn.commit()
            print(f"Inserted {count} rows so far...")

conn.commit()
cur.close()
conn.close()

print(f"Done. Inserted {count} rows into papers table.")