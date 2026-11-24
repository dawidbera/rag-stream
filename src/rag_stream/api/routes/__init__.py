"""Routes package init."""

from rag_stream.api.routes.documents import router as documents_router
from rag_stream.api.routes.health import router as health_router
from rag_stream.api.routes.query import router as query_router

__all__ = ["documents_router", "health_router", "query_router"]
