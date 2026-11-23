"""Hierarchical chunking service implementing Parent-Child strategy."""

from uuid import uuid4

from rag_stream.config import settings
from rag_stream.domain.models import DocumentChunk, ParsedDocument


class HierarchicalChunker:
    """Partitions parsed documents into hierarchical Parent-Child chunks.

    Parent chunks (default ~1000 tokens / ~4000 characters) capture broad context
    suitable for LLM generation.
    Child chunks (default ~250 tokens / ~1000 characters, ~50 tokens overlap) provide
    high-precision semantic targeting for dense vector and sparse keyword search.
    """

    def __init__(
        self,
        child_chunk_size: int = settings.child_chunk_size,
        child_chunk_overlap: int = settings.child_chunk_overlap,
        parent_chunk_size: int = settings.parent_chunk_size,
    ):
        """Initialize chunker parameters in approximate token counts (1 token ~= 4 chars)."""
        self.child_chars = child_chunk_size * 4
        self.child_overlap_chars = child_chunk_overlap * 4
        self.parent_chars = parent_chunk_size * 4

    def chunk_document(self, parsed_doc: ParsedDocument) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        """Generate linked parent and child chunks for a parsed document.

        Args:
            parsed_doc: ParsedDocument with page content.

        Returns:
            Tuple of (parent_chunks, child_chunks).
        """
        parent_chunks: list[DocumentChunk] = []
        child_chunks: list[DocumentChunk] = []
        chunk_idx = 0

        # Process page by page to retain accurate page citations
        for page in parsed_doc.pages:
            page_num = page.get("page_number", 1)
            text = page.get("content", "")
            if not text:
                continue

            # Step 1: Create parent chunks
            parent_slices = self._split_text(text, self.parent_chars, overlap_chars=0)

            for p_slice in parent_slices:
                p_id = f"parent_{uuid4().hex[:8]}"
                p_chunk = DocumentChunk(
                    chunk_id=p_id,
                    document_id=parsed_doc.document_id,
                    parent_id=None,
                    content=p_slice,
                    chunk_index=chunk_idx,
                    is_parent=True,
                    page_number=page_num,
                    token_count=max(1, len(p_slice) // 4),
                    metadata={
                        **parsed_doc.metadata,
                        "filename": parsed_doc.filename,
                    },
                )
                parent_chunks.append(p_chunk)
                chunk_idx += 1

                # Step 2: Create child chunks within each parent
                child_slices = self._split_text(
                    p_slice,
                    self.child_chars,
                    overlap_chars=self.child_overlap_chars,
                )

                for c_slice in child_slices:
                    c_id = f"chunk_{uuid4().hex[:8]}"
                    c_chunk = DocumentChunk(
                        chunk_id=c_id,
                        document_id=parsed_doc.document_id,
                        parent_id=p_id,
                        content=c_slice,
                        chunk_index=chunk_idx,
                        is_parent=False,
                        page_number=page_num,
                        token_count=max(1, len(c_slice) // 4),
                        metadata={
                            **parsed_doc.metadata,
                            "filename": parsed_doc.filename,
                            "parent_id": p_id,
                            "parent_content": p_slice,
                        },
                    )
                    child_chunks.append(c_chunk)
                    chunk_idx += 1

        return parent_chunks, child_chunks

    @staticmethod
    def _split_text(text: str, target_size: int, overlap_chars: int) -> list[str]:
        """Split text into segments of approximately target_size characters with overlap."""
        if len(text) <= target_size:
            return [text.strip()] if text.strip() else []

        step = max(1, target_size - overlap_chars)
        slices: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + target_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                slices.append(chunk)
            if end >= len(text):
                break
            start += step

        return slices
