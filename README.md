Started building a RAG system for Canadian financial regulatory compliance (OSFI, FINTRAC, PIPEDA). 

A few early tradeoffs already worth sharing.

pgvector over a dedicated vector DB. I want hybrid search (BM25 + vector) later without running two separate databases. Postgres + pgvector means one less moving part, at the cost of some raw vector-search speed compared to Pinecone or Qdrant.

Pydantic schemas before any pipeline code. Defining strict data contracts (DocumentRecord, ChunkRecord, GeneratedAnswer) up front feels slower initially, but it means every later node gets validated data instead of raw dicts, catching bugs at the boundary instead of deep in the pipeline.
