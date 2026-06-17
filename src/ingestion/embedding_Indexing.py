import json
import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.ingestion.extract import extract_pdf
from src.ingestion.chunk import chunk_pages

load_dotenv()

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded.")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

#instering into the postgresSQL
def insert_document(cur, record):
    cur.execute("""
        INSERT INTO documents (doc_id, filename, title, source, total_pages)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO NOTHING
    """, (record.doc_id, record.filename, record.title, record.source, record.total_pages))

def insert_chunk(cur, chunk, embedding):
    cur.execute("""
        INSERT INTO chunks (chunk_id, doc_id, page_number, chunk_index, text, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO NOTHING
    """, (
        chunk.chunk_id, chunk.doc_id, chunk.page_number,
        chunk.chunk_index, chunk.text, chunk.token_count,
        embedding.tolist()
    ))


def run_ingestion():
    with open("data/manifest.json", "r") as f:
        manifest = json.load(f)

    conn = get_connection()
    cur = conn.cursor()

    total_chunks = 0

    for doc_id, meta in manifest.items():
        print(f"\nProcessing: {meta['title']}")

        record, pages = extract_pdf(
            doc_id=doc_id,
            file_path=meta["file_path"],
            source=meta["source"],
            title=meta["title"]
        )

        chunks = chunk_pages(doc_id=doc_id, pages=pages)
        print(f"  {len(chunks)} chunks created")

        insert_document(cur, record)

        for chunk in chunks:
            embedding = model.encode(chunk.text)
            insert_chunk(cur, chunk, embedding)

        conn.commit()
        total_chunks += len(chunks)
        print(f"  Inserted into database.")

    cur.close()
    conn.close()
    print(f"\nDone. Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    run_ingestion()