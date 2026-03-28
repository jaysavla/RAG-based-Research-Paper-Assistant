import faiss
import numpy as np
import store


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return store.EMBED_MODEL.encode(texts, show_progress_bar=False, batch_size=32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    vectors = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
