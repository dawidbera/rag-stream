"""Health check endpoint routes."""

from fastapi import APIRouter

from rag_stream import __version__
from rag_stream.api.schemas import HealthResponse
from rag_stream.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application health status and component readiness."""
    return HealthResponse(
        status="UP",
        environment=settings.environment,
        version=__version__,
        components={
            "vector_store": "ready",
            "embeddings": "ready",
            "reranker": "ready",
        },
    )
