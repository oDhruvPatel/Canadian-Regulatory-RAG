from pypdf import PdfReader
from pathlib import Path
from ..models.schemas import DocumentRecord
from datetime import datetime

def extract_pdf(doc_id: str, file_path: str, source: str, title: str):
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader):
        text = page.extract_text() or ""
        pages.append({
            "page_number": i + 1,
            "text": text.strip()
        })

    record = DocumentRecord(
        doc_id=doc_id,
        filename=Path(file_path).name,
        title=title,
        source=source,
        total_pages=len(pages),
        ingested_at=datetime.utcnow()
    )

    return record, pages