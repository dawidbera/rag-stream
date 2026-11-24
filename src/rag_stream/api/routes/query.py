"""RAG query and streaming endpoints."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag_stream.api.schemas import QueryRequest, QueryResponse, SourceCitationSchema
from rag_stream.domain.models import QueryRequestModel
from rag_stream.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/v1/query", tags=["Query"])
retrieval_service = RetrievalService()


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Execute standard RAG question-answering with citations and CRAG evaluation."""
    domain_request = QueryRequestModel(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        enable_reranking=request.enable_reranking,
        enable_query_rewrite=request.enable_query_rewrite,
        stream=request.stream,
    )

    response = await retrieval_service.execute_query(domain_request)

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        sources=[
            SourceCitationSchema(
                document_id=s.document_id,
                filename=s.filename,
                chunk_id=s.chunk_id,
                score=s.score,
                content_excerpt=s.content_excerpt,
                page_number=s.page_number,
            )
            for s in response.sources
        ],
        latency_ms=response.latency_ms,
    )


@router.post("/stream")
async def query_documents_stream(request: QueryRequest) -> StreamingResponse:
    """Execute RAG question-answering streaming tokens and citations via Server-Sent Events (SSE)."""
    domain_request = QueryRequestModel(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        enable_reranking=request.enable_reranking,
        enable_query_rewrite=request.enable_query_rewrite,
        stream=True,
    )

    return StreamingResponse(
        retrieval_service.execute_query_stream(domain_request),
        media_type="text/event-stream",
    )
