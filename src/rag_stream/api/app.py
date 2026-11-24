"""FastAPI application factory and middleware configuration."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_stream import __version__
from rag_stream.api.routes.documents import router as documents_router
from rag_stream.api.routes.health import router as health_router
from rag_stream.api.routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown management."""
    # Yield control to the application
    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="RAG Stream",
        description="Production-grade Advanced RAG Engine with Corrective RAG (CRAG), "
                    "hybrid search (Dense + BM25), hierarchical chunking, and streaming citations.",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Route Handlers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_router)

    return app


app = create_app()
