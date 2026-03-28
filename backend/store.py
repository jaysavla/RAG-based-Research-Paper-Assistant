import logging
import os

import faiss
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

logger = logging.getLogger("rag")

# ── OpenAI client ─────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── ML models (loaded once at process start) ──────────────────────────────────
logger.info("Loading embedding model...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")
logger.info("Loading cross-encoder re-ranker...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
logger.info("Models ready.")

# ── In-memory document store ──────────────────────────────────────────────────
DOCUMENT_STORE: dict[str, dict] = {}  # filename → {chunks, embeddings, index}

# ── Global FAISS index (all docs merged) ──────────────────────────────────────
GLOBAL_INDEX: faiss.Index | None = None
GLOBAL_CHUNK_MAP: list[dict] = []  # [{filename, chunk_id, text, pages}, ...]

# ── BM25 keyword index (parallel to GLOBAL_CHUNK_MAP) ────────────────────────
BM25_INDEX: BM25Okapi | None = None
BM25_CORPUS: list[list[str]] = []

# ── Gold evaluation set ───────────────────────────────────────────────────────
EVAL_SET: list[dict] = []

# ── Async upload job tracker ──────────────────────────────────────────────────
JOBS: dict[str, dict] = {}

# ── Documents uploaded in the current process session (not restored from disk) ─
SESSION_DOCS: set[str] = set()
