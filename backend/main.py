from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
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


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        content = await file.read()
        pdf_bytes = io.BytesIO(content)

        pages = extract_text_by_page(pdf_bytes)
        num_pages = len(pages)
        full_text_length = sum(len(p["text"]) for p in pages)

        chunks = split_into_chunks(pages)

        print(f"[EMBED] Embedding {len(chunks)} chunks for '{file.filename}'...")
        embeddings = embed_chunks(chunks)

        index = build_faiss_index(embeddings)

        DOCUMENT_STORE[file.filename] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "index": index,
        }

        print(f"[UPLOAD] {file.filename} | pages={num_pages} | chunks={len(chunks)} | embed_dim={embeddings.shape[1]}")
        print(f"[FAISS] Index for '{file.filename}': {index.ntotal} vectors indexed")

        results.append({
            "filename": file.filename,
            "num_pages": num_pages,
            "text_length": full_text_length,
            "num_chunks": len(chunks),
            "embed_dim": embeddings.shape[1],
            "vectors_indexed": index.ntotal,
            "chunks": chunks,
        })

    rebuild_global_index()
    return {"uploaded": len(results), "files": results}


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
