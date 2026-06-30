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

def bm25_search(query: str, top_k: int = TOP_K_RETRIEVE):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT c.chunk_id, c.doc_id, c.page_number, c.text,
            d.filename, d.source
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
                """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    corpus = [row[3].split() for row in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())

    scored_rows = list(zip(rows, scores))

    def get_score(item):
        return item[1]
    
    scored_rows.sort(key=get_score, reverse=True)
    top_rows = scored_rows[:top_k]

    results = []
    for row, score in top_rows:
        results.append({
            "chunk_id": row[0],
            "doc_id": row[1],
            "page_number": row[2],
            "text": row[3],
            "filename": row[4],
            "source": row[5],
            "bm25_score": float(score)
        })
    return results

def rrf_fusion(vector_results, bm25_results, k: int = 60):
    """Combine vector + BM25 rankings using Reciprocal Rank Fusion."""
    scores = {}
    chunk_data = {}

    for rank, item in enumerate(vector_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_data[cid] = item

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = item

    fused = []
    for cid, score in scores.items():
        chunk = chunk_data[cid]
        chunk["rrf_score"] = score
        fused.append(chunk)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused

def retrieve(query: str, top_k: int = TOP_K_RETRIEVE):
    """Main entry point: hybrid retrieval combining vector + BM25."""
    vector_results = vector_search(query, top_k)
    bm25_results = bm25_search(query, top_k)
    fused_results = rrf_fusion(vector_results, bm25_results)
    return fused_results[:top_k]