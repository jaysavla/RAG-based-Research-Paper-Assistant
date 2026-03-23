import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Research Assistant", layout="centered")
st.title("RAG Research Paper Assistant")
st.caption("Upload research papers to get started.")

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if uploaded_files:
    overwrite = st.checkbox("Overwrite if already uploaded", value=False)
    if st.button("Process PDFs"):
        files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                data={"overwrite": str(overwrite).lower()},
            )
            response.raise_for_status()
            job_data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure it is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        if "error" in job_data:
            st.error(job_data["error"])
            for r in job_data.get("rejected", []):
                st.caption(f"❌ {r['filename']}: {r['reason']}")
            st.stop()

        if job_data.get("rejected"):
            for r in job_data["rejected"]:
                st.warning(f"❌ Rejected before queuing — **{r['filename']}**: {r['reason']}")

        st.session_state.job_id = job_data["job_id"]
        st.session_state.upload_result = None

# ── Poll until done ───────────────────────────────────────────────────────────
if st.session_state.job_id and st.session_state.upload_result is None:
    job_id = st.session_state.job_id
    try:
        poll = requests.get(f"{BACKEND_URL}/upload/status/{job_id}").json()
    except Exception as e:
        st.error(f"Polling error: {e}")
        st.stop()

    if poll["status"] == "processing":
        st.info(f"Processing... {poll['progress']}")
        time.sleep(2)
        st.rerun()
    elif poll["status"] == "failed":
        st.error(f"Upload failed: {poll['error']}")
        st.session_state.job_id = None
    elif poll["status"] == "done":
        st.session_state.upload_result = poll["result"]
        st.session_state.job_id = None

# ── Show results once done ────────────────────────────────────────────────────
if st.session_state.upload_result:
    data = st.session_state.upload_result
    st.success(f"Processed {data['uploaded']} file(s)")
    st.divider()

    for file_info in data["files"]:
        status = file_info.get("status", "ok")

        if status == "skipped":
            st.warning(f"⏭️ **{file_info['filename']}** — skipped: {file_info.get('skip_reason', '')}")
            continue

        st.subheader(file_info["filename"])

        if status == "warning":
            for w in file_info.get("warnings", []):
                st.warning(f"⚠️ {w}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pages", file_info["num_pages"])
        col2.metric("Chunks", file_info.get("num_chunks", "—"))
        col3.metric("Embed dim", file_info.get("embed_dim", "—"))
        col4.metric("Vectors indexed", file_info.get("vectors_indexed", "—"))

        with st.expander("Preview chunks"):
            for chunk in file_info.get("chunks", [])[:5]:
                st.markdown(f"**Chunk {chunk['chunk_id']}** — {chunk['word_count']} words | pages {chunk['pages']}")
                st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                st.divider()

            if file_info.get("num_chunks", 0) > 5:
                st.info(f"... and {file_info['num_chunks'] - 5} more chunks")

        st.divider()

# ── Query section ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Ask a question across your papers")

query = st.text_input("Enter your question", placeholder="e.g. What are the limitations of these approaches?")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

col_search, col_ask = st.columns(2)
search_clicked = col_search.button("Search (raw chunks)", use_container_width=True)
ask_clicked = col_ask.button("Ask AI (summary + citations)", use_container_width=True)

if search_clicked and query.strip():
    with st.spinner("Searching..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure it is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if "error" in data:
        st.warning(data["error"])
    else:
        st.success(f"Top {len(data['results'])} results for: *{data['query']}*")
        for res in data["results"]:
            with st.expander(f"#{res['rank']} | Score: {res['score']} | {res['filename']} | pages {res['pages']}"):
                st.write(res["text"])

if ask_clicked and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure it is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if "error" in data:
        st.warning(data["error"])
    else:
        st.markdown(data["answer"])
        st.divider()
        st.caption("**Sources used:**")
        for src in data["sources"]:
            st.caption(f"{src['label']} {src['filename']} — pages {src['pages']} | rerank: {src['rerank_score']} | faiss: {src['faiss_score']}")

# ── Evaluation section ────────────────────────────────────────────────────────
st.divider()
with st.expander("Retrieval Evaluation (Recall@k & MRR)", expanded=False):
    st.caption("Generate a gold question set from your uploaded papers, then measure FAISS vs Re-ranking quality.")

    samples = st.slider("Questions per document", min_value=2, max_value=10, value=5)
    eval_k  = st.slider("k (top-k to evaluate)", min_value=1, max_value=10, value=5)

    col_gen, col_run = st.columns(2)
    gen_clicked = col_gen.button("Generate Eval Set", use_container_width=True)
    run_clicked = col_run.button("Run Evaluation", use_container_width=True)

    if gen_clicked:
        with st.spinner("Generating questions via GPT..."):
            try:
                r = requests.post(f"{BACKEND_URL}/generate-eval-set", json={"samples_per_doc": samples})
                r.raise_for_status()
                gdata = r.json()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if "error" in gdata:
            st.warning(gdata["error"])
        else:
            st.success(f"Generated {gdata['eval_set_size']} questions")
            for q in gdata["questions"]:
                st.markdown(f"- **{q['filename']} chunk {q['chunk_id']}:** {q['question']}")
                st.caption(f"Source: {q['source_text'][:200]}...")

    if run_clicked:
        with st.spinner("Running retrieval evaluation..."):
            try:
                r = requests.post(f"{BACKEND_URL}/evaluate", json={"k": eval_k})
                r.raise_for_status()
                edata = r.json()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if "error" in edata:
            st.warning(edata["error"])
        else:
            st.subheader(f"Results — {edata['num_questions']} questions, k={edata['k']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"FAISS Recall@{edata['k']}",  edata["faiss_recall_at_k"])
            col2.metric(f"Rerank Recall@{edata['k']}", edata["rerank_recall_at_k"],
                        delta=round(edata["recall_improvement"], 4))
            col3.metric("FAISS MRR",  edata["faiss_mrr"])
            col4.metric("Rerank MRR", edata["rerank_mrr"],
                        delta=round(edata["mrr_improvement"], 4))

            st.divider()
            st.caption("**Per-question breakdown:**")
            for d in edata["details"]:
                faiss_icon  = "✅" if d["faiss_hit"]  else "❌"
                rerank_icon = "✅" if d["rerank_hit"] else "❌"
                rank_info = (
                    f"FAISS rank {d['faiss_rank'] or '—'} → Rerank rank {d['rerank_rank'] or '—'}"
                )
                st.markdown(
                    f"{faiss_icon}→{rerank_icon} **{d['filename']}** chunk {d['correct_chunk_id']} | "
                    f"{rank_info}"
                )
                st.caption(d["question"])
