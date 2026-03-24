import logging
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import store
from utils import rrf_merge, tokenize  # noqa: F401  (re-exported for legacy callers)

logger = logging.getLogger("rag")


def rebuild_global_index() -> None:
    """Merge all per-document embeddings into one global FAISS index."""
    store.GLOBAL_CHUNK_MAP.clear()
    all_vectors = []

    for filename, doc in store.DOCUMENT_STORE.items():
        for chunk in doc["chunks"]:
            store.GLOBAL_CHUNK_MAP.append({
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "text":     chunk["text"],
                "pages":    chunk["pages"],
            })
        all_vectors.append(doc["embeddings"])

    if not all_vectors:
        return

    merged = np.vstack(all_vectors).astype(np.float32)
    faiss.normalize_L2(merged)
    store.GLOBAL_INDEX = faiss.IndexFlatIP(merged.shape[1])
    store.GLOBAL_INDEX.add(merged)
    logger.info("FAISS global index: %d vectors from %d doc(s)",
                store.GLOBAL_INDEX.ntotal, len(store.DOCUMENT_STORE))
    rebuild_bm25_index()


def rebuild_bm25_index() -> None:
    """Build BM25 keyword index over the same corpus as GLOBAL_CHUNK_MAP."""
    if not store.GLOBAL_CHUNK_MAP:
        return
    store.BM25_CORPUS = [tokenize(c["text"]) for c in store.GLOBAL_CHUNK_MAP]
    store.BM25_INDEX  = BM25Okapi(store.BM25_CORPUS)
    logger.info("BM25 index built: %d documents", len(store.BM25_CORPUS))


