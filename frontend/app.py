import json
import time

import requests
import streamlit as st

from components import chunk_card, load_css

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(load_css(), unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("job_id", None),
    ("upload_result", None),
    ("eval_questions", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar — Upload & Settings ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Research Assistant")
    st.caption("Upload PDFs, ask questions, evaluate retrieval quality.")
    st.divider()

    st.markdown("### Upload Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    overwrite   = st.checkbox("Overwrite existing uploads", value=False)
    process_btn = st.button(
        "Process PDFs", type="primary", use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_btn and uploaded_files:
        files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
        try:
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                data={"overwrite": str(overwrite).lower()},
            )
            resp.raise_for_status()
            job_data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Upload error: {e}")
            st.stop()

        if "error" in job_data:
            st.error(job_data["error"])
            for r in job_data.get("rejected", []):
                st.caption(f"❌ {r['filename']}: {r['reason']}")
        else:
            for r in job_data.get("rejected", []):
                st.warning(f"Skipped **{r['filename']}**: {r['reason']}")
            st.session_state.job_id = job_data["job_id"]
            st.session_state.upload_result = None

    # ── Poll upload job ───────────────────────────────────────────────────────
    if st.session_state.job_id and st.session_state.upload_result is None:
        try:
            poll = requests.get(f"{BACKEND_URL}/upload/status/{st.session_state.job_id}").json()
        except Exception as e:
            st.error(f"Polling error: {e}")
            st.stop()

        if poll["status"] == "processing":
            with st.status(f"{poll['progress']}...", expanded=True):
                st.write("Processing in background — hang tight.")
            time.sleep(2)
            st.rerun()
        elif poll["status"] == "failed":
            st.error(f"Processing failed: {poll['error']}")
            st.session_state.job_id = None
        elif poll["status"] == "done":
            st.session_state.upload_result = poll["result"]
            st.session_state.job_id = None
            st.rerun()

    # ── Upload summary ────────────────────────────────────────────────────────
    if st.session_state.upload_result:
        data = st.session_state.upload_result
        st.success(f"Ready — {data['uploaded']} file(s) indexed")

        for fi in data["files"]:
            if fi.get("status") == "skipped":
                st.warning(f"⏭️ **{fi['filename']}** skipped: {fi.get('skip_reason', '')}")
                continue
            with st.expander(f"📑 {fi['filename']}"):
                c1, c2 = st.columns(2)
                c1.metric("Pages",   fi["num_pages"])
                c2.metric("Chunks",  fi.get("num_chunks", "—"))
                c1.metric("Vectors", fi.get("vectors_indexed", "—"))
                c2.metric("Embed dim", fi.get("embed_dim", "—"))
                for w in fi.get("warnings", []):
                    st.warning(w)

    st.divider()
    st.markdown("### Settings")
    top_k = st.slider("Results to retrieve (top-k)", 1, 15, 6)


# ── Main area — Tabs ──────────────────────────────────────────────────────────
st.markdown("## Research Paper Q&A")

tab_ask, tab_search, tab_chunks, tab_eval = st.tabs(
    ["Ask AI", "Raw Search", "Chunk Preview", "Retrieval Eval"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Ask AI (streaming)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.markdown("Ask a question and get a structured answer with citations from your papers.")
    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the main contributions of these papers?",
        height=80, key="ask_q",
    )
    ask_btn = st.button("Ask AI", type="primary", key="ask_btn")

    if ask_btn and question.strip():
        answer_placeholder = st.empty()
        full_answer = ""

        try:
            with st.spinner("Retrieving and re-ranking context..."):
                resp = requests.post(
                    f"{BACKEND_URL}/ask/stream",
                    json={"query": question, "top_k": top_k},
                    stream=True,
                    timeout=120,
                )
                resp.raise_for_status()

            for raw in resp.iter_content(chunk_size=None, decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("__SOURCES__:"):
                    pass  # sources are used server-side only
                else:
                    full_answer += raw
                    answer_placeholder.markdown(full_answer)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend on port 8000.")
        except Exception as e:
            st.error(f"Streaming error: {e}")

    elif ask_btn:
        st.warning("Please enter a question.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Raw Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("Returns the raw top-k chunks ranked by FAISS + BM25 + reranker — no LLM involved.")
    search_q   = st.text_input("Search query", placeholder="e.g. attention mechanism", key="search_q")
    search_btn = st.button("Search", type="primary", key="search_btn")

    if search_btn and search_q.strip():
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": search_q, "top_k": top_k},
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Search error: {e}")
                st.stop()

        if "error" in data:
            st.warning(data["error"])
        else:
            st.success(f"{len(data['results'])} results for: *{data['query']}*")
            for res in data["results"]:
                with st.expander(
                    f"#{res['rank']}  ·  {res['filename']}  ·  pages {res['pages']}  ·  score {res['score']}"
                ):
                    st.markdown(res["text"])

    elif search_btn:
        st.warning("Please enter a search query.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Chunk Preview
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chunks:
    st.markdown("Browse the first chunks of each uploaded document to verify extraction quality.")

    if st.session_state.upload_result:
        files    = st.session_state.upload_result.get("files", [])
        doc_names = [fi["filename"] for fi in files if fi.get("status") != "skipped"]

        if doc_names:
            selected  = st.selectbox("Select document", doc_names)
            preview_n = st.slider("Chunks to preview", 3, 20, 5)
            doc = next((fi for fi in files if fi["filename"] == selected), None)

            if doc:
                st.caption(f"{doc.get('num_chunks', '?')} total chunks · {doc.get('num_pages', '?')} pages")
                for chunk in doc.get("chunks", [])[:preview_n]:
                    st.markdown(chunk_card(chunk), unsafe_allow_html=True)
        else:
            st.info("No documents available to preview.")
    else:
        st.info("Upload and process PDFs first.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Retrieval Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("Generate a gold question set, then compare FAISS vs Re-ranking vs Hybrid retrieval.")

    col_cfg1, col_cfg2 = st.columns(2)
    num_questions = col_cfg1.number_input("Total questions to generate", min_value=2, max_value=30, value=5)
    eval_k        = col_cfg2.slider("k for Recall@k / MRR", 1, 15, 5)

    col_gen, col_run = st.columns(2)
    gen_btn = col_gen.button("Generate Eval Set", use_container_width=True)
    run_btn = col_run.button("Run Evaluation",    use_container_width=True)

    if gen_btn:
        with st.spinner("Generating questions via GPT..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/generate-eval-set",
                    json={"num_questions": num_questions},
                )
                r.raise_for_status()
                gdata = r.json()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if "error" in gdata:
            st.warning(gdata["error"])
        else:
            st.session_state.eval_questions = gdata["questions"]
            st.success(f"Generated {gdata['eval_set_size']} questions")

    if st.session_state.eval_questions:
        with st.expander(f"View {len(st.session_state.eval_questions)} questions"):
            for q in st.session_state.eval_questions:
                st.markdown(f"**{q['filename']} · chunk {q['chunk_id']}**")
                st.write(q["question"])
                st.caption(q["source_text"][:200] + "...")
                st.divider()

    if run_btn:
        with st.spinner("Evaluating all 3 retrieval pipelines..."):
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
            st.markdown(f"#### Results — {edata['num_questions']} questions, k={edata['k']}")

            st.markdown("**Recall@k**")
            c1, c2, c3 = st.columns(3)
            c1.metric("FAISS",  edata["faiss_recall_at_k"])
            c2.metric("Rerank", edata["rerank_recall_at_k"],
                      delta=round(edata["rerank_recall_at_k"] - edata["faiss_recall_at_k"], 4))
            c3.metric("Hybrid", edata["hybrid_recall_at_k"],
                      delta=round(edata["hybrid_recall_at_k"] - edata["faiss_recall_at_k"], 4))

            st.markdown("**MRR**")
            c4, c5, c6 = st.columns(3)
            c4.metric("FAISS",  edata["faiss_mrr"])
            c5.metric("Rerank", edata["rerank_mrr"],
                      delta=round(edata["rerank_mrr"] - edata["faiss_mrr"], 4))
            c6.metric("Hybrid", edata["hybrid_mrr"],
                      delta=round(edata["hybrid_mrr"] - edata["faiss_mrr"], 4))

            st.divider()
            st.markdown("**Per-question breakdown**")
            for d in edata["details"]:
                fi = "✅" if d["faiss_hit"]  else "❌"
                ri = "✅" if d["rerank_hit"] else "❌"
                hi = "✅" if d["hybrid_hit"] else "❌"
                with st.expander(
                    f"{fi} FAISS · {ri} Rerank · {hi} Hybrid  ·  "
                    f"{d['filename']} chunk {d['correct_chunk_id']}"
                ):
                    st.write(d["question"])
                    cols = st.columns(3)
                    cols[0].metric("FAISS rank",  d["faiss_rank"]  or "—")
                    cols[1].metric("Rerank rank", d["rerank_rank"] or "—")
                    cols[2].metric("Hybrid rank", d["hybrid_rank"] or "—")
