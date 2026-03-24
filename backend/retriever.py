import logging
from typing import Dict, List, Tuple

import faiss
import numpy as np

import store
from config import BGE_QUERY_PREFIX
from indexer import rrf_merge

logger = logging.getLogger("rag")

PROMPT_TEMPLATE = """You are a research assistant. Using ONLY the context below, answer the user's question.

Your response MUST follow this exact structure:

## Summary
A concise answer to the question based on the sources.

## Comparison
How the different sources agree or differ on this topic (skip if only one source).

## Citations
List each source you used, with its label, filename, and page numbers.
Example: [1] paper.pdf, pages [3, 4]

---
Context:
{context}

---
Question: {query}
"""


def faiss_only(query: str, k: int) -> List[int]:
    """Return global CHUNK_MAP indices from FAISS search only."""
    q_vec = store.EMBED_MODEL.encode([BGE_QUERY_PREFIX + query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, idxs = store.GLOBAL_INDEX.search(q_vec, k)
    return [int(i) for i in idxs[0] if i != -1]


def faiss_then_rerank(query: str, k: int) -> List[Tuple[int, float]]:
    """FAISS candidates → cross-encoder re-ranking. Returns (idx, score) pairs."""
    candidates = faiss_only(query, min(k * 3, store.GLOBAL_INDEX.ntotal))
    pairs     = [[query, store.GLOBAL_CHUNK_MAP[i]["text"]] for i in candidates]
    ce_scores = store.RERANKER.predict(pairs)
    ranked    = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [(int(idx), float(score)) for idx, score in ranked[:k]]


def hybrid_then_rerank(query: str, k: int) -> List[Tuple[int, float]]:
    """BM25 + FAISS → RRF merge → cross-encoder re-ranking. Returns (idx, score) pairs."""
    pool       = min(k * 3, store.GLOBAL_INDEX.ntotal)
    faiss_idxs = faiss_only(query, pool)
    tokens     = query.lower().split()
    bm25_idxs  = list(np.argsort(store.BM25_INDEX.get_scores(tokens))[::-1][:pool])
    merged     = rrf_merge(faiss_idxs, bm25_idxs, pool)
    pairs      = [[query, store.GLOBAL_CHUNK_MAP[i]["text"]] for i in merged]
    ce_scores  = store.RERANKER.predict(pairs)
    ranked     = sorted(zip(merged, ce_scores), key=lambda x: x[1], reverse=True)
    return [(int(idx), float(score)) for idx, score in ranked[:k]]


def retrieve_and_build_prompt(query: str, top_k: int) -> Tuple[str, List[Dict]]:
    """Hybrid retrieval → build LLM prompt + sources metadata."""
    ranked = hybrid_then_rerank(query, top_k)

    retrieved = []
    for idx, ce_score in ranked:
        chunk = store.GLOBAL_CHUNK_MAP[idx]
        retrieved.append({**chunk, "rerank_score": round(ce_score, 4)})

    context = "\n\n".join(
        f"[{i+1}] Source: {r['filename']}, pages {r['pages']}\n{r['text']}"
        for i, r in enumerate(retrieved)
    )
    prompt  = PROMPT_TEMPLATE.format(context=context, query=query)
    sources = [
        {"label": f"[{i+1}]", "filename": r["filename"],
         "pages": r["pages"], "rerank_score": r["rerank_score"]}
        for i, r in enumerate(retrieved)
    ]
    return prompt, sources
