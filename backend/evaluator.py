import logging
from typing import Dict, List, Optional

import store
from retriever import faiss_only, faiss_then_rerank, hybrid_then_rerank

logger = logging.getLogger("rag")


def generate_eval_set(samples_per_doc: int) -> List[Dict]:
    """Sample chunks evenly per doc, ask GPT to generate one question per chunk."""
    store.EVAL_SET.clear()

    for filename, doc in store.DOCUMENT_STORE.items():
        good_chunks = [c for c in doc["chunks"] if c["word_count"] >= 50] or doc["chunks"]
        step        = max(1, len(good_chunks) // samples_per_doc)
        sampled     = good_chunks[::step][:samples_per_doc]

        for chunk in sampled:
            prompt = (
                "Read this passage from a research paper and write ONE specific question that:\n"
                "1. Can ONLY be answered using this passage\n"
                "2. Is NOT a yes/no question\n"
                "3. Asks about a specific fact, method, result, or limitation\n\n"
                f"Passage:\n{chunk['text'][:800]}\n\n"
                "Respond with ONLY the question, nothing else."
            )
            resp = store.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            question = resp.choices[0].message.content.strip()
            store.EVAL_SET.append({
                "question":    question,
                "filename":    filename,
                "chunk_id":    chunk["chunk_id"],
                "source_text": chunk["text"][:300],
            })
            logger.info("Eval-gen: '%s' → %s chunk %d", question[:70], filename, chunk["chunk_id"])

    logger.info("Eval-gen complete: %d questions across %d doc(s)",
                len(store.EVAL_SET), len(store.DOCUMENT_STORE))
    return store.EVAL_SET


def run_evaluation(k: int) -> Dict:
    """Run each gold question through 3 pipelines; return Recall@k and MRR."""
    faiss_hits,  rerank_hits,  hybrid_hits  = [], [], []
    faiss_rr,    rerank_rr,    hybrid_rr    = [], [], []
    details = []

    for item in store.EVAL_SET:
        correct: Optional[int] = next(
            (i for i, c in enumerate(store.GLOBAL_CHUNK_MAP)
             if c["filename"] == item["filename"] and c["chunk_id"] == item["chunk_id"]),
            None,
        )
        if correct is None:
            logger.warning("Eval — chunk not found: %s #%d", item["filename"], item["chunk_id"])
            continue

        f_res = faiss_only(item["question"], k)
        r_res = [idx for idx, _ in faiss_then_rerank(item["question"], k)]
        h_res = [idx for idx, _ in hybrid_then_rerank(item["question"], k)]

        f_hit = int(correct in f_res);  r_hit = int(correct in r_res);  h_hit = int(correct in h_res)
        faiss_hits.append(f_hit);  rerank_hits.append(r_hit);  hybrid_hits.append(h_hit)

        f_rank = f_res.index(correct) + 1 if f_hit else None
        r_rank = r_res.index(correct) + 1 if r_hit else None
        h_rank = h_res.index(correct) + 1 if h_hit else None
        faiss_rr.append(1.0 / f_rank if f_rank else 0.0)
        rerank_rr.append(1.0 / r_rank if r_rank else 0.0)
        hybrid_rr.append(1.0 / h_rank if h_rank else 0.0)

        details.append({
            "question": item["question"], "filename": item["filename"],
            "correct_chunk_id": item["chunk_id"],
            "faiss_hit":  bool(f_hit),  "faiss_rank":  f_rank,
            "rerank_hit": bool(r_hit),  "rerank_rank": r_rank,
            "hybrid_hit": bool(h_hit),  "hybrid_rank": h_rank,
        })
        logger.info(
            "Eval — FAISS hit=%d rank=%s | Rerank hit=%d rank=%s | Hybrid hit=%d rank=%s | '%s'",
            f_hit, f_rank, r_hit, r_rank, h_hit, h_rank, item["question"][:50],
        )

    n = len(faiss_hits)
    if n == 0:
        return {"error": "No valid eval items could be matched to the current index."}

    return {
        "k": k, "num_questions": n,
        "faiss_recall_at_k":  round(sum(faiss_hits)  / n, 4),
        "rerank_recall_at_k": round(sum(rerank_hits) / n, 4),
        "hybrid_recall_at_k": round(sum(hybrid_hits) / n, 4),
        "faiss_mrr":          round(sum(faiss_rr)    / n, 4),
        "rerank_mrr":         round(sum(rerank_rr)   / n, 4),
        "hybrid_mrr":         round(sum(hybrid_rr)   / n, 4),
        "details": details,
    }
