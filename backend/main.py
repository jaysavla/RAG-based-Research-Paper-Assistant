from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
import pdfplumber
import io
import re
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="RAG Research Assistant")

CHUNK_SIZE = 300   # target words per chunk
CHUNK_OVERLAP = 50 # words of overlap between chunks

# Load model once at startup — not per request
print("[STARTUP] Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[STARTUP] Model ready.")

# In-memory store: filename -> {chunks, embeddings}
DOCUMENT_STORE: Dict[str, Dict] = {}


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

        # Store in memory for retrieval later
        DOCUMENT_STORE[file.filename] = {
            "chunks": chunks,
            "embeddings": embeddings,
        }

        print(f"[UPLOAD] {file.filename} | pages={num_pages} | chunks={len(chunks)} | embed_dim={embeddings.shape[1]}")

        results.append({
            "filename": file.filename,
            "num_pages": num_pages,
            "text_length": full_text_length,
            "num_chunks": len(chunks),
            "embed_dim": embeddings.shape[1],
            "chunks": chunks,
        })

    return {"uploaded": len(results), "files": results}
