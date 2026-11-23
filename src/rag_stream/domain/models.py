"""Domain entities and value objects for the RAG Stream system."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a raw or uploaded document in the system.

    Attributes:
        document_id: Unique identifier for the document.
        filename: Original file name.
        content_type: MIME type of the document (e.g. application/pdf, text/markdown).
        raw_text: Extracted full text content.
        metadata: Associated document-level metadata (author, created date, etc.).
        created_at: ISO timestamp of ingestion.
    """

    document_id: str = Field(default_factory=lambda: f"doc_{uuid4().hex[:8]}")
    filename: str
    content_type: str = "text/plain"
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ParsedDocument(BaseModel):
    """Intermediate parsed representation containing page-level segments and extracted tables."""

    document_id: str
    filename: str
    full_text: str
    pages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Represents an indexed chunk derived from a document.

    Supports hierarchical chunking (parent-child relationship) where child chunks
    are indexed for high-precision vector search and parent chunks provide broad context.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        document_id: Reference to parent document.
        parent_id: Optional ID of the parent chunk if this is a child chunk.
        content: Textual content of this chunk.
        chunk_index: Order index of this chunk within the document.
        is_parent: Whether this chunk is a parent container.
        page_number: Source page number if available.
        token_count: Approximate token count of the content.
        metadata: Custom chunk metadata.
        dense_vector: Dense embedding vector.
        sparse_vector: Sparse BM25 vector represented as indices and values.
    """

    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid4().hex[:8]}")
    document_id: str
    parent_id: str | None = None
    content: str
    chunk_index: int = 0
    is_parent: bool = False
    page_number: int | None = 1
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, Any] | None = None


class ScoredChunk(BaseModel):
    """A retrieved chunk enriched with multi-stage ranking scores."""

    chunk: DocumentChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class SourceCitation(BaseModel):
    """Attributed citation referencing a specific chunk and document."""

    document_id: str
    filename: str
    chunk_id: str
    score: float
    content_excerpt: str
    page_number: int | None = 1


class QueryRequestModel(BaseModel):
    """Internal domain model for executing a RAG query."""

    query: str
    collection_name: str = "documents"
    top_k: int = 5
    enable_reranking: bool = True
    enable_query_rewrite: bool = True
    stream: bool = False


class QueryResponseModel(BaseModel):
    """Domain model representing a finalized RAG answer with citations."""

    query: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    latency_ms: int = 0
    is_fallback: bool = False
