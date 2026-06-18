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

