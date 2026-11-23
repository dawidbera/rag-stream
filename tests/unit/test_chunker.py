"""Tests for HierarchicalChunker."""

from rag_stream.domain.models import ParsedDocument
from rag_stream.services.chunker import HierarchicalChunker


def test_hierarchical_chunking_parent_child_linking():
    """Ensure child chunks link to their respective parent chunk ID."""
    long_text = "Paragraph test sentence for semantic verification. " * 50
    parsed = ParsedDocument(
        document_id="doc_test_3",
        filename="doc.txt",
        full_text=long_text,
        pages=[{"page_number": 1, "content": long_text}],
    )

    chunker = HierarchicalChunker(
        child_chunk_size=50,
        child_chunk_overlap=10,
        parent_chunk_size=200,
    )
    parents, children = chunker.chunk_document(parsed)

    assert len(parents) > 0
    assert len(children) >= len(parents)

    parent_ids = {p.chunk_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids
        assert child.document_id == "doc_test_3"
        assert child.is_parent is False
        assert "parent_content" in child.metadata
