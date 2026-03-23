import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Research Assistant", layout="centered")
st.title("RAG Research Paper Assistant")
st.caption("Upload research papers to get started.")

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Uploading, chunking and embedding text (first run downloads model ~90MB)..."):
            files = [
                ("files", (f.name, f.read(), "application/pdf"))
                for f in uploaded_files
            ]
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure it is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.success(f"Processed {data['uploaded']} file(s)")
        st.divider()

        for file_info in data["files"]:
            st.subheader(file_info["filename"])

            col1, col2, col3 = st.columns(3)
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

if st.button("Search") and query.strip():
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
