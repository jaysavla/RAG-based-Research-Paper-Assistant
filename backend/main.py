import json
import logging
import uuid

import faiss
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse

# Configure logging before importing modules that use it at load time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import store                                    # noqa: E402  (models load here)
from config import BGE_QUERY_PREFIX
from evaluator import generate_eval_set, run_evaluation
from jobs import process_upload
from models import AskRequest, EvalGenRequest, EvalRunRequest, QueryRequest
from persistence import load_store
from retriever import retrieve_and_build_prompt
from validator import validate_file

logger = logging.getLogger("rag")

app = FastAPI(title="RAG Research Assistant")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    load_store()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG backend is running"}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    overwrite: bool = Form(False),
):
    job_id, file_payloads, rejected = str(uuid.uuid4())[:8], [], []

    for file in files:
        content = await file.read()
        error   = validate_file(content)
        if error:
            rejected.append({"filename": file.filename, "reason": error})
            logger.warning("Rejected '%s': %s", file.filename, error)
        else:
            file_payloads.append({"filename": file.filename, "content": content})

    if not file_payloads:
        return {"error": "All files were rejected.", "rejected": rejected}

    store.JOBS[job_id] = {
        "status": "processing", "progress": "Queued",
        "files":  [f["filename"] for f in file_payloads],
        "rejected": rejected, "result": None, "error": None,
    }
    background_tasks.add_task(process_upload, job_id, file_payloads, overwrite)
    logger.info("Job %s queued — %d valid, %d rejected", job_id, len(file_payloads), len(rejected))

    return {
        "job_id": job_id, "status": "processing",
        "files":  [f["filename"] for f in file_payloads],
        "rejected": rejected, "poll_url": f"/upload/status/{job_id}",
    }


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    if job_id not in store.JOBS:
        return {"error": f"Job '{job_id}' not found."}
    job = store.JOBS[job_id]
    return {
        "job_id": job_id, "status": job["status"], "progress": job["progress"],
        "files":  job["files"],  "result": job["result"], "error": job["error"],
    }


# ── Query (raw chunks) ────────────────────────────────────────────────────────

@app.post("/query")
def query_documents(req: QueryRequest):
    if store.GLOBAL_INDEX is None or store.GLOBAL_INDEX.ntotal == 0:
        return {"error": "No documents uploaded yet. Please upload PDFs first."}

    logger.info("QUERY top_k=%d — '%s'", req.top_k, req.query)
    q_vec = store.EMBED_MODEL.encode([BGE_QUERY_PREFIX + req.query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = store.GLOBAL_INDEX.search(q_vec, req.top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        chunk = store.GLOBAL_CHUNK_MAP[idx]
        results.append({
            "rank": rank + 1, "score": round(float(score), 4),
            "filename": chunk["filename"], "chunk_id": chunk["chunk_id"],
            "pages": chunk["pages"],       "text": chunk["text"],
        })
    return {"query": req.query, "top_k": req.top_k, "results": results}


# ── Ask (non-streaming) ───────────────────────────────────────────────────────

@app.post("/ask")
def ask_documents(req: AskRequest):
    if store.GLOBAL_INDEX is None or store.GLOBAL_INDEX.ntotal == 0:
        return {"error": "No documents uploaded yet. Please upload PDFs first."}

    prompt, sources = retrieve_and_build_prompt(req.query, req.top_k)
    logger.info("ASK — '%s' | %d sources", req.query, len(sources))

    response = store.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    logger.info("ASK — LLM response received (%d chars)", len(answer))
    return {"query": req.query, "answer": answer, "sources": sources}


# ── Ask (streaming) ───────────────────────────────────────────────────────────

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    if store.GLOBAL_INDEX is None or store.GLOBAL_INDEX.ntotal == 0:
        def _err():
            yield "No documents uploaded yet. Please upload PDFs first."
        return StreamingResponse(_err(), media_type="text/plain")

    prompt, sources = retrieve_and_build_prompt(req.query, req.top_k)
    logger.info("ASK/STREAM — '%s' | %d sources", req.query, len(sources))

    def generate():
        yield f"__SOURCES__:{json.dumps(sources)}\n"
        stream = store.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.post("/generate-eval-set")
def generate_eval_set_endpoint(req: EvalGenRequest):
    if not store.SESSION_DOCS:
        return {"error": "No documents uploaded in this session. Please upload PDFs first."}
    questions = generate_eval_set(req.num_questions)
    return {"eval_set_size": len(questions), "questions": questions}


@app.post("/evaluate")
def run_evaluation_endpoint(req: EvalRunRequest):
    if store.GLOBAL_INDEX is None:
        return {"error": "No documents uploaded yet."}
    if not store.EVAL_SET:
        return {"error": "Eval set is empty. Call /generate-eval-set first."}
    return run_evaluation(req.k)
