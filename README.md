# Canadian Financial Regulatory RAG

A Retrieval-Augmented Generation system for querying Canadian financial regulatory documents — with cited, hallucination-controlled answers.

## What It Does

Compliance teams spend significant time manually cross-referencing dense regulatory documents. This system lets you ask questions like:

> "What does OSFI E-23 require before deploying a credit-scoring model, and how does that interact with PIPEDA consent obligations?"

And get a traceable, cited answer in under 2 seconds.

## Tech Stack

- **LangGraph** — query pipeline (rewrite → retrieve → rerank → generate → eval)
- **PostgreSQL + pgvector** — hybrid BM25 + vector search in one database
- **Dagster** — ingestion pipeline with automatic document update detection
- **FastAPI** — REST API with typed Pydantic responses
- **Streamlit** — simple frontend UI
- **RAGAS + DeepEval** — evaluation with CI/CD quality gate

## Document Corpus

| Document | Source |
|---|---|
| OSFI Guideline E-23 (Model Risk Management) | osfi-bsif.gc.ca |
| Treasury Board Directive on Automated Decision-Making | tbs-sct.canada.ca |
| PIPEDA (key sections) | priv.gc.ca |
| OSFI Guideline B-13 (Technology & Cyber Risk) | osfi-bsif.gc.ca |
| Algorithmic Impact Assessment Tool | github.com/canada-ca |


## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- Cohere API key (free tier)

### 1. Clone and configure
```bash
git clone https://github.com/yourusername/canadian-regulatory-rag.git
cd canadian-regulatory-rag
cp .env.example .env
```

Fill in your API keys in `.env`:
```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 2. Start all services
```bash
docker compose up -d
```

### 3. Run the ingestion pipeline
```bash
docker compose exec api python -m src.ingestion.run_pipeline
```

