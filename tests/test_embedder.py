import faiss
import numpy as np

from embedder import build_faiss_index


def test_returns_faiss_index():
    vecs = np.random.rand(10, 128).astype(np.float32)
    idx = build_faiss_index(vecs)
    assert isinstance(idx, faiss.Index)


def test_ntotal_equals_number_of_input_vectors():
    vecs = np.random.rand(7, 64).astype(np.float32)
    idx = build_faiss_index(vecs)
    assert idx.ntotal == 7


def test_index_dimension_matches_input():
    dim = 256
    vecs = np.random.rand(3, dim).astype(np.float32)
    idx = build_faiss_index(vecs)
    assert idx.d == dim


def test_cosine_similarity_scores_at_most_one():
    # Vectors are L2-normalised internally; cosine sim of any query ≤ 1.0
    vecs = np.random.rand(10, 32).astype(np.float32)
    idx = build_faiss_index(vecs)
    q = np.random.rand(1, 32).astype(np.float32)
    faiss.normalize_L2(q)
    scores, _ = idx.search(q, 1)
    assert float(scores[0][0]) <= 1.01  # small float tolerance


def test_identical_vector_returns_score_one():
    # A vector searched against itself should return cosine sim ≈ 1.0
    vecs = np.random.rand(5, 32).astype(np.float32)
    idx = build_faiss_index(vecs)
    q = vecs[0:1].copy()
    faiss.normalize_L2(q)
    scores, indices = idx.search(q, 1)
    assert indices[0][0] == 0
    assert abs(float(scores[0][0]) - 1.0) < 1e-5


def test_single_vector_index():
    vecs = np.random.rand(1, 64).astype(np.float32)
    idx = build_faiss_index(vecs)
    assert idx.ntotal == 1
