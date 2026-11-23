"""Application configuration and settings using Pydantic Settings."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration class for RAG Stream application.

    Defines environment variables, defaults, threshold settings,
    and external adapter endpoints.
    """

    # Application Settings
    app_name: str = "rag-stream"
    environment: str = "production"
    debug: bool = False
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000

    # Vector Store (Qdrant) Configuration
    qdrant_url: Optional[str] = None  # None indicates in-memory client
    qdrant_api_key: Optional[str] = None
    default_collection: str = "documents"
    dense_vector_dim: int = 384  # Matches default dense embedding dimension

    # Ingestion & Chunking Parameters
    child_chunk_size: int = 250
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1000

    # Retrieval & Search Parameters
    default_top_k: int = 5
    rerank_candidate_pool: int = 25
    rrf_constant_k: int = 60

    # Corrective RAG (CRAG) Parameters
    crag_min_confidence_threshold: float = 0.35
    fallback_message: str = (
        "I could not find sufficiently reliable or relevant context in the uploaded "
        "documents to answer your question accurately. Please refine or expand your query."
    )

    # LLM & Generation Configuration
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
