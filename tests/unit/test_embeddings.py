"""Unit tests for dense and sparse embedding adapters."""

from rag_stream.adapters.embeddings import BM25SparseTokenizer, DenseEmbeddingAdapter


def test_dense_embeddings_normalization_and_dimension():
    """Verify dense embeddings produce normalized vectors of target dimension."""
    adapter = DenseEmbeddingAdapter(dimension=128)
    vec = adapter.embed_text("Cloud infrastructure cost management in Kubernetes")

    assert len(vec) == 128
    # Norm should be approximately 1.0
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_bm25_sparse_tokenizer_indices_and_weights():
    """Verify BM25 tokenizer generates valid positive indices and frequency weights."""
    tokenizer = BM25SparseTokenizer()
    sparse = tokenizer.tokenize_sparse("Kubernetes cost scaling and Kubernetes cluster")

    assert "indices" in sparse and "values" in sparse
    assert len(sparse["indices"]) == len(sparse["values"])
    assert len(sparse["indices"]) > 0
    # Every index must be a positive integer
    assert all(i > 0 for i in sparse["indices"])
    # Values should be non-negative
    assert all(v > 0 for v in sparse["values"])
