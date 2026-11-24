"""Unit tests for CorrectiveRAGEvaluator."""

from rag_stream.domain.models import DocumentChunk, ScoredChunk
from rag_stream.services.crag_evaluator import CorrectiveRAGEvaluator


def test_crag_evaluator_triggers_fallback_when_below_threshold():
    """Ensure CRAG evaluator marks low confidence results as irrelevant with fallback."""
    evaluator = CorrectiveRAGEvaluator(min_threshold=0.40)

    chunk = DocumentChunk(chunk_id="c_low", document_id="doc_1", content="Unrelated topic")
    candidates = [ScoredChunk(chunk=chunk, rerank_score=0.25, final_score=0.25)]

    verdict = evaluator.evaluate(candidates)

    assert verdict.is_relevant is False
    assert verdict.highest_confidence == 0.25
    assert len(verdict.selected_chunks) == 0
    assert verdict.fallback_message is not None


def test_crag_evaluator_approves_relevant_context():
    """Ensure CRAG evaluator approves relevant candidates above threshold."""
    evaluator = CorrectiveRAGEvaluator(min_threshold=0.40)

    chunk1 = DocumentChunk(chunk_id="c_high", document_id="doc_1", content="Relevant cloud costs")
    chunk2 = DocumentChunk(chunk_id="c_border", document_id="doc_1", content="Marginally relevant")

    candidates = [
        ScoredChunk(chunk=chunk1, rerank_score=0.85, final_score=0.85),
        ScoredChunk(chunk=chunk2, rerank_score=0.45, final_score=0.45),
    ]

    verdict = evaluator.evaluate(candidates)

    assert verdict.is_relevant is True
    assert verdict.highest_confidence == 0.85
    assert len(verdict.selected_chunks) == 2
    assert verdict.fallback_message is None
