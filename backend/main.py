from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import uuid
import pdfplumber
import io
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="RAG Research Assistant")

CHUNK_SIZE = 200   # target words per chunk
CHUNK_OVERLAP = 50 # words of overlap between chunks

MAX_FILE_MB       = 50
MAX_FILE_BYTES    = MAX_FILE_MB * 1024 * 1024
MIN_TEXT_CHARS    = 200   # below this → likely image-only or nearly blank
PDF_MAGIC         = b"%PDF"

# Load models once at startup — not per request
print("[STARTUP] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("allenai/specter")
print("[STARTUP] Loading cross-encoder re-ranker...")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[STARTUP] Models ready.")

# In-memory store: filename -> {chunks, embeddings, index}
DOCUMENT_STORE: Dict[str, Dict] = {}

# Global index across all documents for multi-doc retrieval
# Stores (filename, chunk_id) so we can trace back any result
GLOBAL_INDEX = None
GLOBAL_CHUNK_MAP: List[Dict] = []  # [{filename, chunk_id, text, pages}, ...]

# Gold evaluation set: [{question, filename, chunk_id, source_text}, ...]
EVAL_SET: List[Dict] = []

# Upload jobs: job_id -> {status, progress, files, result, error}
JOBS: Dict[str, Dict] = {}


def validate_file(content: bytes) -> Optional[str]:
    """Return an error string if the file should be rejected, else None."""
    if len(content) == 0:
        return "File is empty (0 bytes)."
    if len(content) > MAX_FILE_BYTES:
        return f"File exceeds {MAX_FILE_MB} MB limit ({len(content) // (1024*1024)} MB)."
    if not content.startswith(PDF_MAGIC):
        return "File does not appear to be a valid PDF (missing %PDF header)."
    return None


def extract_text_by_page(pdf_bytes: io.BytesIO) -> List[Dict]:
    pages = []
    with pdfplumber.open(pdf_bytes) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def clean_text(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r'\d+', stripped):  # lone page number
            continue
        cleaned.append(stripped)
    return ' '.join(cleaned)


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
    sentence_pages = []
    for page_data in pages:
        text = clean_text(page_data["text"])
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                sentence_pages.append((sent, page_data["page"]))

    if not sentence_pages:
        return []

    chunks = []
    chunk_id = 0
    i = 0

    while i < len(sentence_pages):
        chunk_sentences = []
        chunk_pages = set()
        word_count = 0

        j = i
        while j < len(sentence_pages) and word_count < CHUNK_SIZE:
            sent, pg = sentence_pages[j]
            chunk_sentences.append(sent)
            chunk_pages.add(pg)
            word_count += len(sent.split())
            j += 1

        chunk_text = ' '.join(chunk_sentences)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "word_count": word_count,
            "char_count": len(chunk_text),
            "pages": sorted(chunk_pages),
        })
        chunk_id += 1

        words_to_skip = max(1, word_count - CHUNK_OVERLAP)
        skipped = 0
        while i < len(sentence_pages) and skipped < words_to_skip:
            skipped += len(sentence_pages[i][0].split())
            i += 1

    return chunks


def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False, batch_size=32)
    return embeddings  # shape: (num_chunks, 384)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a cosine-similarity FAISS index (normalize + inner product)."""
    dim = embeddings.shape[1]
    vectors = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine similarity
    index.add(vectors)
    return index


def rebuild_global_index():
    """Merge all per-document indexes into one global index for cross-doc search."""
    global GLOBAL_INDEX, GLOBAL_CHUNK_MAP

    GLOBAL_CHUNK_MAP = []
    all_vectors = []

    for filename, doc in DOCUMENT_STORE.items():
        for chunk in doc["chunks"]:
            GLOBAL_CHUNK_MAP.append({
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "pages": chunk["pages"],
            })
        all_vectors.append(doc["embeddings"])

    if not all_vectors:
        return

    merged = np.vstack(all_vectors).astype(np.float32)
    faiss.normalize_L2(merged)
    dim = merged.shape[1]
    GLOBAL_INDEX = faiss.IndexFlatIP(dim)
    GLOBAL_INDEX.add(merged)
    print(f"[FAISS] Global index rebuilt: {GLOBAL_INDEX.ntotal} vectors from {len(DOCUMENT_STORE)} doc(s)")


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG backend is running"}


def _process_upload(job_id: str, file_payloads: List[Dict], overwrite: bool = False):
    """Background task: extract → chunk → embed → index. Updates JOBS[job_id] throughout."""
    job = JOBS[job_id]
    results = []
    processed = 0

    try:
        for payload in file_payloads:
            filename = payload["filename"]
            content  = payload["content"]
            warnings = []
            file_status = "ok"

            # ── Duplicate check ───────────────────────────────────────────────
            if filename in DOCUMENT_STORE and not overwrite:
                print(f"[JOB {job_id}] Skipping duplicate: '{filename}'")
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "skip_reason": "Already uploaded. Enable overwrite to re-process.",
                    "warnings": [],
                })
                continue

            # ── Extract text ──────────────────────────────────────────────────
            job["progress"] = f"Extracting text: {filename}"
            try:
                pages = extract_text_by_page(io.BytesIO(content))
            except Exception as exc:
                err = str(exc).lower()
                reason = (
                    "PDF is password-protected." if "password" in err or "encrypt" in err
                    else f"Could not parse PDF: {exc}"
                )
                print(f"[JOB {job_id}] Skipping '{filename}': {reason}")
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "skip_reason": reason,
                    "warnings": [],
                })
                continue

            num_pages_total = len(pages)

            # ── Image-only / scanned detection ────────────────────────────────
            if num_pages_total == 0:
                print(f"[JOB {job_id}] Skipping '{filename}': no extractable text (image-only or scanned)")
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "skip_reason": "No text could be extracted. This PDF may be image-only or scanned.",
                    "warnings": [],
                })
                continue

            full_text_length = sum(len(p["text"]) for p in pages)

            # ── Insufficient text warning ─────────────────────────────────────
            if full_text_length < MIN_TEXT_CHARS:
                warnings.append(
                    f"Very little text extracted ({full_text_length} chars). "
                    "Results may be poor — check if the PDF is mostly images."
                )
                file_status = "warning"

            # ── Blank-page warning ────────────────────────────────────────────
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                total_pdf_pages = len(pdf.pages)
            blank_pages = total_pdf_pages - num_pages_total
            if blank_pages > 0:
                warnings.append(f"{blank_pages} page(s) had no extractable text and were skipped.")

            chunks = split_into_chunks(pages)
            if not chunks:
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "skip_reason": "Chunking produced no output — document may be too short.",
                    "warnings": warnings,
                })
                continue

            # ── Embed & index ─────────────────────────────────────────────────
            job["progress"] = f"Embedding {len(chunks)} chunks: {filename}"
            print(f"[JOB {job_id}] Embedding {len(chunks)} chunks for '{filename}'")
            embeddings = embed_chunks(chunks)

            job["progress"] = f"Building FAISS index: {filename}"
            index = build_faiss_index(embeddings)

            DOCUMENT_STORE[filename] = {"chunks": chunks, "embeddings": embeddings, "index": index}
            processed += 1

            print(f"[JOB {job_id}] Done '{filename}' | pages={num_pages_total} chunks={len(chunks)} warnings={warnings}")
            results.append({
                "filename": filename,
                "status": file_status,
                "warnings": warnings,
                "num_pages": num_pages_total,
                "text_length": full_text_length,
                "num_chunks": len(chunks),
                "embed_dim": int(embeddings.shape[1]),
                "vectors_indexed": index.ntotal,
                "chunks": chunks,
            })

        job["progress"] = "Rebuilding global index"
        rebuild_global_index()

        job["status"]   = "done"
        job["progress"] = "Complete"
        job["result"]   = {"uploaded": processed, "files": results}
        print(f"[JOB {job_id}] Finished — {processed} processed, {len(results) - processed} skipped")

    except Exception as exc:
        job["status"] = "failed"
        job["error"]  = str(exc)
        print(f"[JOB {job_id}] FAILED: {exc}")


@app.post("/upload")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    overwrite: bool = Form(False),
):
    """Validate files immediately, return job_id, process in background."""
    job_id = str(uuid.uuid4())[:8]

    file_payloads = []
    rejected = []

    for file in files:
        content = await file.read()
        error = validate_file(content)
        if error:
            rejected.append({"filename": file.filename, "reason": error})
            print(f"[UPLOAD] Rejected '{file.filename}': {error}")
        else:
            file_payloads.append({"filename": file.filename, "content": content})

    if not file_payloads:
        return {"error": "All files were rejected.", "rejected": rejected}

    JOBS[job_id] = {
        "status":   "processing",
        "progress": "Queued",
        "files":    [f["filename"] for f in file_payloads],
        "rejected": rejected,
        "result":   None,
        "error":    None,
    }

    background_tasks.add_task(_process_upload, job_id, file_payloads, overwrite)
    print(f"[UPLOAD] Job {job_id} queued — {len(file_payloads)} valid, {len(rejected)} rejected")

    return {
        "job_id":   job_id,
        "status":   "processing",
        "files":    [f["filename"] for f in file_payloads],
        "rejected": rejected,
        "poll_url": f"/upload/status/{job_id}",
    }


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    if job_id not in JOBS:
        return {"error": f"Job '{job_id}' not found."}
    job = JOBS[job_id]
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
        "files":    job["files"],
        "result":   job["result"],
        "error":    job["error"],
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/query")
def query_documents(req: QueryRequest):
    if GLOBAL_INDEX is None or GLOBAL_INDEX.ntotal == 0:
        return {"error": "No documents uploaded yet. Please upload PDFs first."}

    print(f"[QUERY] '{req.query}' | top_k={req.top_k}")

    query_vec = EMBED_MODEL.encode([req.query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = GLOBAL_INDEX.search(query_vec, req.top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        chunk = GLOBAL_CHUNK_MAP[idx]
        results.append({
            "rank": rank + 1,
            "score": round(float(score), 4),
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "pages": chunk["pages"],
            "text": chunk["text"],
        })
        print(f"  #{rank+1} score={score:.4f} | {chunk['filename']} chunk {chunk['chunk_id']} pages {chunk['pages']}")

    return {"query": req.query, "top_k": req.top_k, "results": results}


class AskRequest(BaseModel):
    query: str
    top_k: int = 8


@app.post("/ask")
def ask_documents(req: AskRequest):
    if GLOBAL_INDEX is None or GLOBAL_INDEX.ntotal == 0:
        return {"error": "No documents uploaded yet. Please upload PDFs first."}

    # Step 1: FAISS retrieval — fetch 3x candidates for re-ranking
    faiss_k = min(req.top_k * 3, GLOBAL_INDEX.ntotal)
    query_vec = EMBED_MODEL.encode([req.query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = GLOBAL_INDEX.search(query_vec, faiss_k)

    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        chunk = GLOBAL_CHUNK_MAP[idx]
        candidates.append({**chunk, "faiss_score": round(float(score), 4)})

    # Step 2: Cross-encoder re-ranking
    pairs = [[req.query, c["text"]] for c in candidates]
    ce_scores = RERANKER.predict(pairs)
    for c, ce_score in zip(candidates, ce_scores):
        c["rerank_score"] = round(float(ce_score), 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    retrieved = candidates[: req.top_k]
    print(f"[RERANK] {len(candidates)} candidates → top {len(retrieved)} after re-ranking")

    # Build context block with citation labels [1], [2], ...
    context_lines = []
    for i, r in enumerate(retrieved):
        label = f"[{i+1}]"
        context_lines.append(
            f"{label} Source: {r['filename']}, pages {r['pages']}\n{r['text']}"
        )
    context_block = "\n\n".join(context_lines)

    print(f"[ASK] '{req.query}' | {len(retrieved)} chunks sent to LLM")

    prompt = f"""You are a research assistant. Using ONLY the context below, answer the user's question.

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
{context_block}

---
Question: {req.query}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    print(f"[ASK] LLM response received ({len(answer)} chars)")

    return {
        "query": req.query,
        "answer": answer,
        "sources": [
            {"label": f"[{i+1}]", "filename": r["filename"], "pages": r["pages"],
             "faiss_score": r["faiss_score"], "rerank_score": r["rerank_score"]}
            for i, r in enumerate(retrieved)
        ],
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _faiss_only(query: str, k: int) -> List[int]:
    """Return global CHUNK_MAP indices from FAISS search, no re-ranking."""
    q_vec = EMBED_MODEL.encode([query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, idxs = GLOBAL_INDEX.search(q_vec, k)
    return [int(i) for i in idxs[0] if i != -1]


def _faiss_then_rerank(query: str, k: int) -> List[int]:
    """Return global CHUNK_MAP indices after FAISS + cross-encoder re-ranking."""
    faiss_k = min(k * 3, GLOBAL_INDEX.ntotal)
    q_vec = EMBED_MODEL.encode([query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, idxs = GLOBAL_INDEX.search(q_vec, faiss_k)
    candidates = [int(i) for i in idxs[0] if i != -1]
    pairs = [[query, GLOBAL_CHUNK_MAP[i]["text"]] for i in candidates]
    ce_scores = RERANKER.predict(pairs)
    ranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]


# ── Evaluation endpoints ──────────────────────────────────────────────────────

class EvalGenRequest(BaseModel):
    samples_per_doc: int = 5


@app.post("/generate-eval-set")
def generate_eval_set(req: EvalGenRequest):
    """Sample chunks from each uploaded doc, ask GPT to generate one question per chunk."""
    if not DOCUMENT_STORE:
        return {"error": "No documents uploaded yet."}

    global EVAL_SET
    EVAL_SET = []

    for filename, doc in DOCUMENT_STORE.items():
        # Only use chunks with enough content
        good_chunks = [c for c in doc["chunks"] if c["word_count"] >= 50]
        if not good_chunks:
            good_chunks = doc["chunks"]

        # Spread samples evenly across the document
        step = max(1, len(good_chunks) // req.samples_per_doc)
        sampled = good_chunks[::step][: req.samples_per_doc]

        for chunk in sampled:
            prompt = (
                "Read this passage from a research paper and write ONE specific question that:\n"
                "1. Can ONLY be answered using this passage\n"
                "2. Is NOT a yes/no question\n"
                "3. Asks about a specific fact, method, result, or limitation\n\n"
                f"Passage:\n{chunk['text'][:800]}\n\n"
                "Respond with ONLY the question, nothing else."
            )
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            question = resp.choices[0].message.content.strip()
            EVAL_SET.append({
                "question": question,
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"][:300],
            })
            print(f"[EVAL-GEN] '{question[:70]}' → {filename} chunk {chunk['chunk_id']}")

    print(f"[EVAL-GEN] Gold set: {len(EVAL_SET)} questions across {len(DOCUMENT_STORE)} doc(s)")
    return {"eval_set_size": len(EVAL_SET), "questions": EVAL_SET}


class EvalRunRequest(BaseModel):
    k: int = 5


@app.post("/evaluate")
def run_evaluation(req: EvalRunRequest):
    """Run every gold question through FAISS-only and FAISS+Rerank. Report Recall@k and MRR."""
    if GLOBAL_INDEX is None:
        return {"error": "No documents uploaded yet."}
    if not EVAL_SET:
        return {"error": "Eval set is empty. Call /generate-eval-set first."}

    k = req.k
    faiss_hits, rerank_hits = [], []
    faiss_rr, rerank_rr = [], []
    details = []

    for item in EVAL_SET:
        # Find the correct chunk's position in the global map
        correct_global_idx: Optional[int] = next(
            (i for i, c in enumerate(GLOBAL_CHUNK_MAP)
             if c["filename"] == item["filename"] and c["chunk_id"] == item["chunk_id"]),
            None,
        )
        if correct_global_idx is None:
            print(f"[EVAL] Skipping — chunk not found in global map: {item['filename']} #{item['chunk_id']}")
            continue

        f_results = _faiss_only(item["question"], k)
        r_results = _faiss_then_rerank(item["question"], k)

        f_hit = int(correct_global_idx in f_results)
        r_hit = int(correct_global_idx in r_results)
        faiss_hits.append(f_hit)
        rerank_hits.append(r_hit)

        f_rank = f_results.index(correct_global_idx) + 1 if f_hit else None
        r_rank = r_results.index(correct_global_idx) + 1 if r_hit else None
        faiss_rr.append(1.0 / f_rank if f_rank else 0.0)
        rerank_rr.append(1.0 / r_rank if r_rank else 0.0)

        details.append({
            "question": item["question"],
            "filename": item["filename"],
            "correct_chunk_id": item["chunk_id"],
            "faiss_hit": bool(f_hit),
            "rerank_hit": bool(r_hit),
            "faiss_rank": f_rank,
            "rerank_rank": r_rank,
        })
        print(
            f"[EVAL] FAISS hit={f_hit} rank={f_rank} | Rerank hit={r_hit} rank={r_rank} | "
            f"'{item['question'][:55]}'"
        )

    n = len(faiss_hits)
    if n == 0:
        return {"error": "No valid eval items could be matched to the current index."}

    return {
        "k": k,
        "num_questions": n,
        "faiss_recall_at_k":  round(sum(faiss_hits)  / n, 4),
        "rerank_recall_at_k": round(sum(rerank_hits) / n, 4),
        "faiss_mrr":          round(sum(faiss_rr)    / n, 4),
        "rerank_mrr":         round(sum(rerank_rr)   / n, 4),
        "recall_improvement": round((sum(rerank_hits) - sum(faiss_hits)) / n, 4),
        "mrr_improvement":    round((sum(rerank_rr)   - sum(faiss_rr))   / n, 4),
        "details": details,
    }
