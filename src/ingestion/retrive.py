import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 20))

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def vector_search(query: str, top_k: int = TOP_K_RETRIEVE):
    query_embedding = model.encode(query).tolist()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
           d.filename, d.source,
           1 - (c.embedding <=> %s::vector) as similarity
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
    """, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []

    for row in rows:
        results.append({
            "chunk_id": row[0],
            "doc_id": row[1],
            "page_number": row[2],
            "text": row[3],
            "filename": row[4],
            "source": row[5],
            "vector_score": float(row[6])
        })
    return results
