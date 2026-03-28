import logging
from typing import Dict, List, Optional, Set

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import os

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
DOCUMENT_STORE: Dict[str, Dict] = {}          # filename → {chunks, embeddings, index}

# ── Global FAISS index (all docs merged) ──────────────────────────────────────
GLOBAL_INDEX: Optional[faiss.Index] = None
GLOBAL_CHUNK_MAP: List[Dict] = []             # [{filename, chunk_id, text, pages}, ...]

# ── BM25 keyword index (parallel to GLOBAL_CHUNK_MAP) ────────────────────────
BM25_INDEX: Optional[BM25Okapi] = None
BM25_CORPUS: List[List[str]] = []

# ── Gold evaluation set ───────────────────────────────────────────────────────
EVAL_SET: List[Dict] = []

# ── Async upload job tracker ──────────────────────────────────────────────────
JOBS: Dict[str, Dict] = {}

# ── Documents uploaded in the current process session (not restored from disk) ─
SESSION_DOCS: Set[str] = set()
