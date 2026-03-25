# RAG Research Paper Assistant

A production-structured Retrieval-Augmented Generation (RAG) system for querying multiple research papers. Built from scratch without LangChain — every component of the retrieval pipeline is implemented and evaluated directly.

---

## What It Does

Upload research PDFs and ask questions across them. The system retrieves the most relevant passages using a hybrid pipeline, re-ranks them with a cross-encoder, and generates a structured answer with citations via GPT-4o-mini.

---

## Retrieval Pipeline

```
PDF Upload
    │
    ▼
spaCy Sentence-Aware Chunking
    │
    ▼
BGE-large-en-v1.5 Embeddings  ──────────────────────────┐
    │                                                     │
    ▼                                                     ▼
FAISS Dense Search                              BM25 Keyword Search
(semantic similarity)                           (stopword-filtered)
    │                                                     │
    └──────────────── RRF Fusion ─────────────────────────┘
                           │
                           ▼
              CrossEncoder Re-ranking
              (ms-marco-MiniLM-L-6-v2)
                           │
                           ▼
              GPT-4o-mini — Structured Answer
              (Summary · Comparison · Citations)
```

---

## Features

- **Hybrid Retrieval** — BM25 + FAISS merged via Reciprocal Rank Fusion
- **Cross-Encoder Re-ranking** — passage-level scoring after candidate retrieval
- **spaCy Chunking** — sentence-boundary-aware, handles abbreviations correctly
- **Retrieval Evaluation** — Recall@k and MRR measured across 3 pipelines (FAISS-only, Rerank, Hybrid)
- **Streaming Responses** — token-by-token LLM output via FastAPI `StreamingResponse`
- **Async Ingestion** — background job processing with polling endpoint
- **Persistent Index** — FAISS + embeddings saved to disk; reloads on restart with dimension validation
- **Edge Case Handling** — empty files, size limits, image-only PDFs, corrupted PDFs, duplicates

---

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI routes
│   ├── store.py         # Global state & ML model singletons
│   ├── config.py        # Constants
│   ├── chunker.py       # PDF extraction + spaCy chunking
│   ├── embedder.py      # BGE embeddings + FAISS index builder
│   ├── indexer.py       # Global FAISS + BM25 index management
│   ├── retriever.py     # FAISS-only / Rerank / Hybrid pipelines
│   ├── evaluator.py     # Recall@k & MRR evaluation
│   ├── jobs.py          # Async upload background task
│   ├── persistence.py   # Save/load store to disk
│   ├── validator.py     # File validation
│   ├── models.py        # Pydantic request models
│   └── utils.py         # RRF merge + BM25 tokenizer
│
├── frontend/
│   ├── app.py           # Streamlit UI logic
│   ├── components.py    # HTML component builders
│   └── style.css        # UI styles
│
└── tests/
    ├── conftest.py
    ├── test_utils.py
    ├── test_validator.py
    ├── test_chunker.py
    ├── test_models.py
    ├── test_embedder.py
    └── test_persistence.py
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set your OpenAI API key

Create a `.env` file in the root:

```
OPENAI_API_KEY=sk-...
```

### 4. Start the backend

```bash
# Windows
.venv\Scripts\Activate.ps1
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the frontend

```bash
# Windows
.venv\Scripts\Activate.ps1
cd frontend
streamlit run app.py
```
---

## Evaluation

The built-in evaluation tab compares three retrieval strategies on a fixed question set:

| Pipeline | Recall@5 | MRR |
|---|---|---|
| FAISS-only | — | — |
| FAISS + Rerank | — | — |
| Hybrid (BM25+FAISS+RRF+Rerank) | — | — |

*Run the eval on your own papers and fill in the numbers.*

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDFs (async, returns job_id) |
| `GET` | `/upload/status/{job_id}` | Poll upload job status |
| `POST` | `/query` | Raw chunk search (no LLM) |
| `POST` | `/ask` | Non-streaming answer |
| `POST` | `/ask/stream` | Streaming answer |
| `POST` | `/generate-eval-set` | Generate gold question set |
| `POST` | `/evaluate` | Run Recall@k + MRR evaluation |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11 |
| Embeddings | BGE-large-en-v1.5 (SentenceTransformers) |
| Re-ranker | ms-marco-MiniLM-L-6-v2 (CrossEncoder) |
| Vector Search | FAISS (IndexFlatIP) |
| Keyword Search | BM25Okapi (rank-bm25) |
| PDF Extraction | pdfplumber |
| Sentence Splitting | spaCy (en_core_web_sm) |
| LLM | GPT-4o-mini (OpenAI) |
| Frontend | Streamlit |
| Testing | pytest |
