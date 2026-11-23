"""Integration tests for Qdrant vector store adapter in in-memory mode."""

from rag_stream.adapters.vector_store import QdrantVectorStoreAdapter
from rag_stream.domain.models import DocumentChunk


def test_vector_store_upsert_and_search_dense():
    """Test upserting chunks and performing dense vector search in Qdrant."""
    adapter = QdrantVectorStoreAdapter(url=None, dimension=8)
    collection = "test_col"

    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        content="Kubernetes costs scaling",
        dense_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        sparse_vector={"indices": [10, 20], "values": [1.5, 2.0]},
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        document_id="d1",
        content="Database backup procedures",
        dense_vector=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        sparse_vector={"indices": [30, 40], "values": [1.0, 1.0]},
    )

    adapter.upsert_chunks(collection, [chunk1, chunk2])

    results = adapter.search_dense(
        collection_name=collection,
        query_vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        limit=2,
    )

    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].dense_score > 0.9


def test_vector_store_delete_document():
    """Test document deletion removes its corresponding chunks."""
    adapter = QdrantVectorStoreAdapter(url=None, dimension=8)
    collection = "test_delete_col"

    chunk = DocumentChunk(
        chunk_id="c_del",
        document_id="doc_to_delete",
        content="Temporary content",
        dense_vector=[0.5] * 8,
    )
    adapter.upsert_chunks(collection, [chunk])

    # Verify present
    res = adapter.search_dense(collection, [0.5] * 8, limit=1)
    assert len(res) == 1

    # Delete
    deleted = adapter.delete_document(collection, "doc_to_delete")
    assert deleted is True

    # Verify gone
    res_after = adapter.search_dense(collection, [0.5] * 8, limit=1)
    assert len(res_after) == 0
