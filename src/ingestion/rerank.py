import os
import cohere
from dotenv import load_dotenv
from src.models.schemas import RankedChunk

load_dotenv()

TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))


def rerank(query: str, chunks: list[dict]) -> list[RankedChunk]:
    """Rerank retrieved chunks using Cohere rerank API."""

    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]

    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=TOP_K_RERANK
    )

    results = []
    for item in response.results:
        chunk = chunks[item.index]
        results.append(RankedChunk(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            filename=chunk["filename"],
            source=chunk["source"],
            page_number=chunk["page_number"],
            text=chunk["text"],
            rerank_score=float(item.relevance_score)
        ))

    return results