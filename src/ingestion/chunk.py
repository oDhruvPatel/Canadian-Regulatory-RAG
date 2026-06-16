from src.models.schemas import ChunkRecord
import uuid

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def chunk_pages(doc_id: str, pages: list[dict]) -> list[ChunkRecord]:
    records = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        words = text.split()
        start = 0

        while start < len(words):
            chunk_words = words[start: start + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)

            records.append(ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                page_number=page["page_number"],
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_words)
            ))

            chunk_index += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return records