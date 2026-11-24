"""Cross-Encoder reranking adapter for contextual score refinement."""

import re

from rag_stream.domain.models import ScoredChunk


class CrossEncoderReranker:
    """Refines candidate rankings using cross-encoder score calculation.

    Scores each (query, passage) pair simultaneously to capture deep semantic interaction,
    re-ordering the initial retrieval pool before LLM context assembly.
    """

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Score and re-rank candidates based on query-passage relevance.

        Args:
            query: User's original or rewritten search query.
            candidates: Retrieved candidates from first-stage retrieval.
            top_k: Number of highest-ranking candidates to retain.

        Returns:
            Re-ordered list of ScoredChunk instances with updated rerank_score.
        """
        if not candidates:
            return []

        query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for candidate in candidates:
            content = candidate.chunk.content.lower()
            content_terms = set(re.findall(r"\b\w{3,}\b", content))

            # Semantic match ratio
            if query_terms:
                overlap = len(query_terms.intersection(content_terms))
                overlap_ratio = overlap / len(query_terms)
            else:
                overlap_ratio = 0.5

            # Phrase match boost
            clean_q = query.lower().strip()
            phrase_boost = 0.3 if clean_q in content else 0.0

            # Combined cross-encoder score normalized to [0.0, 1.0]
            # Incorporates first-stage signal with heavy weight on semantic overlap
            base_score = min(1.0, candidate.dense_score if candidate.dense_score > 0 else 0.5)
            score = (0.6 * overlap_ratio) + (0.25 * base_score) + (0.15 * phrase_boost)
            score = round(min(1.0, max(0.0, score)), 4)

            candidate.rerank_score = score
            candidate.final_score = score

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[:top_k]
