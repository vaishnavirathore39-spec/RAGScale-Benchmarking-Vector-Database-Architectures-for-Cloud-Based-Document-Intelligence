import psycopg2
from sentence_transformers import SentenceTransformer

DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Rit123&&",
    "host": "localhost",
    "port": "5432"
}

model = SentenceTransformer('all-MiniLM-L6-v2')

query = "graph neural networks for recommendation systems"
query_embedding = model.encode(query).tolist()

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

cur.execute(
    """
    SELECT title, categories, embedding <-> %s::vector AS distance
    FROM papers
    ORDER BY embedding <-> %s::vector
    LIMIT 5
    """,
    (query_embedding, query_embedding)
)

results = cur.fetchall()
print(f"\nTop 5 results for: '{query}'\n")
for title, categories, distance in results:
    print(f"[{categories}] {title.strip()}  (distance: {distance:.4f})")

cur.close()
conn.close()