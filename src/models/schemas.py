
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# Ingestion models

class DocumentRecord(BaseModel):
    """Row written to the documents table after PDF registration."""
    doc_id: str
    filename: str
    title: str
    source: str  # OSFI | FINTRAC | PIPEDA
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    total_pages: int
    total_chunks: int = 0


class ChunkRecord(BaseModel):
    """Row written to the chunks table for each text chunk."""
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int



# Retrieval models

class RetrievedChunk(BaseModel):
    """A chunk returned from vector/BM25 search, before reranking."""
    chunk_id: str
    doc_id: str
    filename: str
    source: str
    page_number: int
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0

class RankedChunk(BaseModel):
    """A chunk after cross-encoder reranking."""
    chunk_id: str
    doc_id: str
    filename: str
    source: str
    page_number: int
    text: str
    rerank_score: float

# Citation & Answer models

class Citation(BaseModel):
    """One cited source attached to the generated answer."""
    chunk_id: str
    filename: str
    source: str
    page_number: int
    excerpt: str

class GeneratedAnswer(BaseModel):
    """Structured output expected from Gemini."""
    answer: str
    citations: list[Citation]
    confidence: str  # HIGH | MEDIUM | LOW
    disclaimer: str = (
        "This response is for informational purposes only and does not "
        "constitute legal or compliance advice. Always consult a qualified "
        "professional for regulatory guidance."
    )


# LangGraph pipeline state

class PipelineState(BaseModel):
    """Shared state passed between every node in the LangGraph pipeline."""

    # Input
    original_query: str

    # Cache node
    cache_hit: bool = False
    cached_response: Optional[RAGResponse] = None

    # Query rewrite node
    rewritten_query: Optional[str] = None

    # Retrieve node
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)

    # Rerank node
    ranked_chunks: list[RankedChunk] = Field(default_factory=list)

    # Generate node
    raw_answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)

    # Inline eval node
    eval_score: Optional[int] = None
    eval_reasoning: Optional[str] = None

    # Citation check node
    citations_valid: bool = False

    # Retry control
    retry_count: int = 0
    max_retries: int = 2

    # Final output
    final_response: Optional[RAGResponse] = None

    class Config:
        arbitrary_types_allowed = True



# Evaluation models


class EvalResult(BaseModel):
    """Output from the inline LLM-as-judge node."""
    score: int  # 1-5
    reasoning: str
    pass_threshold: bool  # True if score >= 3


class EvalDatasetItem(BaseModel):
    """One row in eval/eval_dataset.json."""
    question: str
    expected_keywords: list[str]
    source_doc: str
    difficulty: str = "medium"



# Cache model

class CacheEntry(BaseModel):
    """Row written to the query_cache table."""
    cache_id: str
    query_text: str
    response_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    hit_count: int = 0



# API response model

class RAGResponse(BaseModel):
    """Top-level JSON returned by the FastAPI /query endpoint."""
    question: str
    answer: str
    citations: list[Citation]
    confidence: str
    eval_score: Optional[int] = None
    disclaimer: str
    cached: bool = False
    latency_ms: Optional[float] = None


PipelineState.model_rebuild()