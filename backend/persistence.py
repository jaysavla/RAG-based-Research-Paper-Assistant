import json
import logging
import os
import re

import faiss
import numpy as np

import store
from config import STORE_DIR

logger = logging.getLogger("rag")


def _safe_name(filename: str) -> str:
    return re.sub(r"[^\w\-.]", "_", filename)


def save_store() -> None:
    """Persist GLOBAL_INDEX, GLOBAL_CHUNK_MAP, and per-doc data to disk."""
    if store.GLOBAL_INDEX is None or store.GLOBAL_INDEX.ntotal == 0:
        return

    os.makedirs(STORE_DIR, exist_ok=True)
    faiss.write_index(store.GLOBAL_INDEX, os.path.join(STORE_DIR, "global.index"))

    with open(os.path.join(STORE_DIR, "chunk_map.json"), "w") as f:
        json.dump(store.GLOBAL_CHUNK_MAP, f)

    docs_dir = os.path.join(STORE_DIR, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    for filename, doc in store.DOCUMENT_STORE.items():
        doc_dir = os.path.join(docs_dir, _safe_name(filename))
        os.makedirs(doc_dir, exist_ok=True)
        with open(os.path.join(doc_dir, "chunks.json"), "w") as f:
            json.dump({"filename": filename, "chunks": doc["chunks"]}, f)
        np.save(os.path.join(doc_dir, "embeddings.npy"), doc["embeddings"])

    logger.info("Saved %d doc(s) to '%s/'", len(store.DOCUMENT_STORE), STORE_DIR)


def load_store() -> None:
    """Load persisted data from disk into memory on startup."""
    from embedder import build_faiss_index
    from indexer import rebuild_bm25_index

    index_path     = os.path.join(STORE_DIR, "global.index")
    chunk_map_path = os.path.join(STORE_DIR, "chunk_map.json")
    docs_dir       = os.path.join(STORE_DIR, "docs")

    if not os.path.exists(index_path):
        logger.info("No saved store found — starting fresh.")
        return

    # Reject persisted data if the embedding model changed (dimension mismatch)
    expected_dim = store.EMBED_MODEL.get_sentence_embedding_dimension()
    saved_index  = faiss.read_index(index_path)
    if saved_index.d != expected_dim:
        logger.warning(
            "Persisted index has dim=%d but current model outputs dim=%d — "
            "discarding saved store. Re-upload your documents.",
            saved_index.d, expected_dim,
        )
        return

    store.GLOBAL_INDEX = saved_index

    with open(chunk_map_path) as f:
        store.GLOBAL_CHUNK_MAP[:] = json.load(f)

    if os.path.exists(docs_dir):
        for safe in os.listdir(docs_dir):
            doc_dir         = os.path.join(docs_dir, safe)
            chunks_path     = os.path.join(doc_dir, "chunks.json")
            embeddings_path = os.path.join(doc_dir, "embeddings.npy")
            if not (os.path.exists(chunks_path) and os.path.exists(embeddings_path)):
                continue
            with open(chunks_path) as f:
                doc_data = json.load(f)
            filename   = doc_data["filename"]
            chunks     = doc_data["chunks"]
            embeddings = np.load(embeddings_path)
            store.DOCUMENT_STORE[filename] = {
                "chunks": chunks, "embeddings": embeddings,
                "index": build_faiss_index(embeddings),
            }

    rebuild_bm25_index()
    logger.info("Loaded %d doc(s), %d vectors.",
                len(store.DOCUMENT_STORE), store.GLOBAL_INDEX.ntotal)
