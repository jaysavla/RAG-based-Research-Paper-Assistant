import io
import logging
from typing import Dict, List

import pdfplumber

import store
from chunker import extract_text_by_page, split_into_chunks
from config import MIN_TEXT_CHARS
from embedder import build_faiss_index, embed_chunks
from indexer import rebuild_global_index
from persistence import save_store

logger = logging.getLogger("rag")


def process_upload(job_id: str, file_payloads: List[Dict], overwrite: bool = False) -> None:
    """Background task: extract → chunk → embed → index → persist."""
    job       = store.JOBS[job_id]
    results   = []
    processed = 0

    try:
        for payload in file_payloads:
            filename    = payload["filename"]
            content     = payload["content"]
            warnings    = []
            file_status = "ok"

            # ── Duplicate check ───────────────────────────────────────────
            if filename in store.DOCUMENT_STORE and not overwrite:
                logger.warning("Job %s — skipping duplicate: '%s'", job_id, filename)
                results.append({"filename": filename, "status": "skipped",
                                "skip_reason": "Already uploaded. Enable overwrite to re-process.",
                                "warnings": []})
                continue

            # ── Extract text ──────────────────────────────────────────────
            job["progress"] = f"Extracting text: {filename}"
            try:
                pages = extract_text_by_page(io.BytesIO(content))
            except Exception as exc:
                err    = str(exc).lower()
                reason = ("PDF is password-protected."
                          if "password" in err or "encrypt" in err
                          else f"Could not parse PDF: {exc}")
                logger.warning("Job %s — skipping '%s': %s", job_id, filename, reason)
                results.append({"filename": filename, "status": "skipped",
                                "skip_reason": reason, "warnings": []})
                continue

            num_pages_total = len(pages)

            # ── Image-only / scanned ──────────────────────────────────────
            if num_pages_total == 0:
                logger.warning("Job %s — skipping '%s': no extractable text", job_id, filename)
                results.append({"filename": filename, "status": "skipped",
                                "skip_reason": "No text could be extracted. PDF may be image-only or scanned.",
                                "warnings": []})
                continue

            full_text_length = sum(len(p["text"]) for p in pages)

            # ── Insufficient text warning ─────────────────────────────────
            if full_text_length < MIN_TEXT_CHARS:
                warnings.append(f"Very little text extracted ({full_text_length} chars). "
                                 "Results may be poor — PDF may be mostly images.")
                file_status = "warning"

            # ── Blank-page warning ────────────────────────────────────────
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                total_pdf_pages = len(pdf.pages)
            blank_pages = total_pdf_pages - num_pages_total
            if blank_pages > 0:
                warnings.append(f"{blank_pages} page(s) had no extractable text and were skipped.")

            # ── Chunk ─────────────────────────────────────────────────────
            chunks = split_into_chunks(pages)
            if not chunks:
                results.append({"filename": filename, "status": "skipped",
                                "skip_reason": "Chunking produced no output — document may be too short.",
                                "warnings": warnings})
                continue

            # ── Embed & index ─────────────────────────────────────────────
            job["progress"] = f"Embedding {len(chunks)} chunks: {filename}"
            logger.info("Job %s — embedding %d chunks for '%s'", job_id, len(chunks), filename)
            embeddings = embed_chunks(chunks)

            job["progress"] = f"Building FAISS index: {filename}"
            index = build_faiss_index(embeddings)

            store.DOCUMENT_STORE[filename] = {
                "chunks": chunks, "embeddings": embeddings, "index": index
            }
            store.SESSION_DOCS.add(filename)
            processed += 1
            logger.info("Job %s — done '%s' | pages=%d chunks=%d warnings=%d",
                        job_id, filename, num_pages_total, len(chunks), len(warnings))

            results.append({
                "filename":       filename,
                "status":         file_status,
                "warnings":       warnings,
                "num_pages":      num_pages_total,
                "text_length":    full_text_length,
                "num_chunks":     len(chunks),
                "embed_dim":      int(embeddings.shape[1]),
                "vectors_indexed": index.ntotal,
                "chunks":         chunks,
            })

        # ── Rebuild indexes & persist ─────────────────────────────────────
        job["progress"] = "Rebuilding global index"
        rebuild_global_index()
        save_store()

        job["status"]  = "done"
        job["progress"] = "Complete"
        job["result"]  = {"uploaded": processed, "files": results}
        logger.info("Job %s — finished: %d processed, %d skipped",
                    job_id, processed, len(results) - processed)

    except Exception as exc:
        job["status"] = "failed"
        job["error"]  = str(exc)
        logger.error("Job %s — FAILED: %s", job_id, exc, exc_info=True)
