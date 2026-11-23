"""Reciprocal Rank Fusion (RRF) algorithm implementation."""

from rag_stream.config import settings
from rag_stream.domain.models import ScoredChunk


class ReciprocalRankFusion:
    """Combines multiple ranked lists into a single unified ranking using RRF.

    Mathematical formula:
        RRF(d) = sum_{m in M} ( 1 / (k + r_m(d)) )
    where:
        - M is the set of rankers (dense ANN search, sparse BM25)
        - r_m(d) is the 1-based rank index of document d in ranker m
        - k is a smoothing constant (default: 60)
    """

    def __init__(self, k: int = settings.rrf_constant_k):
        """Initialize RRF with smoothing constant k."""
        self.k = k

    def fuse(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
        top_k: int = 25,
    ) -> list[ScoredChunk]:
        """Fuse dense and sparse retrieval result sets into a unified candidate list.

        Args:
            dense_results: Ordered list of ScoredChunks from dense vector search.
            sparse_results: Ordered list of ScoredChunks from sparse BM25 search.
            top_k: Maximum number of fused candidates to return.

        Returns:
            List of ScoredChunk instances sorted descending by rrf_score.
        """
        # Map chunk_id to fused ScoredChunk entry
        chunk_map: dict[str, ScoredChunk] = {}
        rrf_scores: dict[str, float] = {}

        # 1. Process dense rankings
        for rank_1_based, item in enumerate(dense_results, start=1):
            c_id = item.chunk.chunk_id
            if c_id not in chunk_map:
                chunk_map[c_id] = item
                rrf_scores[c_id] = 0.0
            else:
                chunk_map[c_id].dense_score = item.dense_score

            rrf_scores[c_id] += 1.0 / (self.k + rank_1_based)

        # 2. Process sparse rankings
        for rank_1_based, item in enumerate(sparse_results, start=1):
            c_id = item.chunk.chunk_id
            if c_id not in chunk_map:
                chunk_map[c_id] = item
                rrf_scores[c_id] = 0.0
            else:
                chunk_map[c_id].sparse_score = item.sparse_score

            rrf_scores[c_id] += 1.0 / (self.k + rank_1_based)

        # 3. Assign computed RRF scores and sort descending
        fused_candidates: list[ScoredChunk] = []
        for c_id, scored_chunk in chunk_map.items():
            scored_chunk.rrf_score = round(rrf_scores[c_id], 6)
            scored_chunk.final_score = scored_chunk.rrf_score
            fused_candidates.append(scored_chunk)

        fused_candidates.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused_candidates[:top_k]
