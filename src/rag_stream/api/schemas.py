"""API Schemas for request and response validation."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful document upload and indexing."""

    document_id: str
    filename: str
    chunks_created: int
    status: str = "INDEXED"


class DeleteDocumentResponse(BaseModel):
    """Response returned after deleting a document."""

    document_id: str
    status: str = "DELETED"


class QueryRequest(BaseModel):
    """Request payload for RAG query endpoints."""

    query: str = Field(..., min_length=3, max_length=1000)
    collection_name: str = "documents"
    top_k: int = Field(default=5, ge=1, le=20)
    enable_reranking: bool = True
    enable_query_rewrite: bool = True
    stream: bool = False


class SourceCitationSchema(BaseModel):
    """Citation details returned with grounded answers."""

    document_id: str
    filename: str
    chunk_id: str
    score: float
    content_excerpt: str
    page_number: int | None = 1


class QueryResponse(BaseModel):
    """Final answer returned to user with citations and latency benchmark."""

    query: str
    answer: str
    sources: list[SourceCitationSchema] = Field(default_factory=list)
    latency_ms: int


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str
    environment: str
    version: str
    components: dict[str, Any]
