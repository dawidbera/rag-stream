"""Unit tests for ReciprocalRankFusion algorithm."""

from rag_stream.domain.models import DocumentChunk, ScoredChunk
from rag_stream.services.rrf import ReciprocalRankFusion


def test_rrf_scoring_mathematical_precision():
    """Verify RRF formula 1/(k + rank) correctly fuses dense and sparse ranked lists."""
    rrf = ReciprocalRankFusion(k=60)

    chunk_a = DocumentChunk(chunk_id="chunk_a", document_id="doc_1", content="Topic A")
    chunk_b = DocumentChunk(chunk_id="chunk_b", document_id="doc_1", content="Topic B")
    chunk_c = DocumentChunk(chunk_id="chunk_c", document_id="doc_1", content="Topic C")

    # Dense ranking: [chunk_a (rank 1), chunk_b (rank 2)]
    dense_results = [
        ScoredChunk(chunk=chunk_a, dense_score=0.9),
        ScoredChunk(chunk=chunk_b, dense_score=0.8),
    ]

    # Sparse ranking: [chunk_b (rank 1), chunk_c (rank 2)]
    sparse_results = [
        ScoredChunk(chunk=chunk_b, sparse_score=15.0),
        ScoredChunk(chunk=chunk_c, sparse_score=10.0),
    ]

    fused = rrf.fuse(dense_results, sparse_results, top_k=3)

    # Expected score calculation:
    # chunk_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.016129 + 0.016393 = 0.032522
    # chunk_a: 1/(60+1) = 1/61 ~= 0.016393
    # chunk_c: 1/(60+2) = 1/62 ~= 0.016129

    assert fused[0].chunk.chunk_id == "chunk_b"
    assert fused[1].chunk.chunk_id == "chunk_a"
    assert fused[2].chunk.chunk_id == "chunk_c"

    assert abs(fused[0].rrf_score - (1.0 / 62.0 + 1.0 / 61.0)) < 1e-4
    assert abs(fused[1].rrf_score - (1.0 / 61.0)) < 1e-4
    assert abs(fused[2].rrf_score - (1.0 / 62.0)) < 1e-4
