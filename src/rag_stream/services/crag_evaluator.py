"""Corrective RAG (CRAG) relevance evaluator and confidence gating."""

from dataclasses import dataclass

from rag_stream.config import settings
from rag_stream.domain.models import ScoredChunk


@dataclass
class CRAGVerdict:
    """Evaluation verdict produced by the CRAG evaluator."""

    is_relevant: bool
    highest_confidence: float
    selected_chunks: list[ScoredChunk]
    fallback_message: str | None = None


class CorrectiveRAGEvaluator:
    """Evaluates retrieved document relevance to prevent hallucinations.

    Implements the Corrective RAG (CRAG) principle:
    - If the highest relevance score meets or exceeds tau_min, context is deemed reliable.
    - If highest score is below tau_min, triggers explicit fallback instead of generating answers
      on hallucinated or irrelevant grounding.
    """

    def __init__(
        self,
        min_threshold: float = settings.crag_min_confidence_threshold,
        fallback_message: str = settings.fallback_message,
    ):
        """Initialize CRAG evaluator with confidence threshold."""
        self.min_threshold = min_threshold
        self.fallback_message = fallback_message

    def evaluate(self, ranked_candidates: list[ScoredChunk]) -> CRAGVerdict:
        """Evaluate candidate pool quality against confidence threshold.

        Args:
            ranked_candidates: Candidates sorted descending by rerank score.

        Returns:
            CRAGVerdict indicating whether context is reliable enough for generation.
        """
        if not ranked_candidates:
            return CRAGVerdict(
                is_relevant=False,
                highest_confidence=0.0,
                selected_chunks=[],
                fallback_message=self.fallback_message,
            )

        highest_score = ranked_candidates[0].rerank_score or ranked_candidates[0].final_score

        if highest_score < self.min_threshold:
            return CRAGVerdict(
                is_relevant=False,
                highest_confidence=highest_score,
                selected_chunks=[],
                fallback_message=self.fallback_message,
            )

        # Retain chunks that meet the minimum threshold
        valid_chunks = [
            c for c in ranked_candidates
            if (c.rerank_score or c.final_score) >= self.min_threshold
        ]

        return CRAGVerdict(
            is_relevant=True,
            highest_confidence=highest_score,
            selected_chunks=valid_chunks,
            fallback_message=None,
        )
