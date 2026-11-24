"""Qdrant vector store adapter supporting dense and sparse hybrid indexing."""

from typing import Any

from qdrant_client import QdrantClient, models

from rag_stream.config import settings
from rag_stream.domain.models import DocumentChunk, ScoredChunk


class QdrantVectorStoreAdapter:
    """Interacts with Qdrant Vector Database.

    Supports in-memory operation (for zero-setup testing and development)
    or remote Qdrant clusters with hybrid sparse + dense vector indices.
    """

    def __init__(
        self,
        url: str | None = settings.qdrant_url,
        api_key: str | None = settings.qdrant_api_key,
        dimension: int = settings.dense_vector_dim,
    ):
        """Initialize client connection to Qdrant."""
        self.dimension = dimension
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            # Use in-memory Qdrant client
            self.client = QdrantClient(location=":memory:")

        self._initialized_collections: set[str] = set()

    def ensure_collection(self, collection_name: str) -> None:
        """Create Qdrant collection with dense and sparse vector configurations if missing."""
        if collection_name in self._initialized_collections:
            return

        collections = [col.name for col in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
        self._initialized_collections.add(collection_name)

    def upsert_chunks(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Upsert document chunks into Qdrant collection with dense and sparse vectors."""
        self.ensure_collection(collection_name)
        if not chunks:
            return

        points: list[models.PointStruct] = []
        for idx, chunk in enumerate(chunks):
            vectors: dict[str, Any] = {}

            if chunk.dense_vector is not None:
                vectors["dense"] = chunk.dense_vector
            else:
                vectors["dense"] = [0.0] * self.dimension

            if chunk.sparse_vector is not None:
                vectors["sparse"] = models.SparseVector(
                    indices=chunk.sparse_vector["indices"],
                    values=chunk.sparse_vector["values"],
                )
            else:
                vectors["sparse"] = models.SparseVector(indices=[1], values=[0.1])

            point_id = abs(hash(chunk.chunk_id)) & 0x7FFFFFFF or (idx + 1)
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "is_parent": chunk.is_parent,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)

    def search_dense(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Perform Approximate Nearest Neighbor (ANN) dense vector search."""
        self.ensure_collection(collection_name)

        # Filter out parent chunks so only child chunks are searched for high precision
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="is_parent",
                    match=models.MatchValue(value=False),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            query_filter=search_filter,
            limit=limit,
        )

        scored_chunks: list[ScoredChunk] = []
        for scored_point in response.points:
            payload = scored_point.payload or {}
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", str(scored_point.id)),
                document_id=payload.get("document_id", ""),
                parent_id=payload.get("parent_id"),
                content=payload.get("content", ""),
                chunk_index=payload.get("chunk_index", 0),
                is_parent=payload.get("is_parent", False),
                page_number=payload.get("page_number", 1),
                token_count=payload.get("token_count", 0),
                metadata=payload.get("metadata", {}),
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    dense_score=float(scored_point.score),
                )
            )

        return scored_chunks

    def search_sparse(
        self,
        collection_name: str,
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Perform sparse BM25 keyword search in Qdrant."""
        self.ensure_collection(collection_name)

        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="is_parent",
                    match=models.MatchValue(value=False),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using="sparse",
            query_filter=search_filter,
            limit=limit,
        )

        scored_chunks: list[ScoredChunk] = []
        for scored_point in response.points:
            payload = scored_point.payload or {}
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", str(scored_point.id)),
                document_id=payload.get("document_id", ""),
                parent_id=payload.get("parent_id"),
                content=payload.get("content", ""),
                chunk_index=payload.get("chunk_index", 0),
                is_parent=payload.get("is_parent", False),
                page_number=payload.get("page_number", 1),
                token_count=payload.get("token_count", 0),
                metadata=payload.get("metadata", {}),
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    sparse_score=float(scored_point.score),
                )
            )

        return scored_chunks

    def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete all child and parent chunks belonging to a document_id."""
        self.ensure_collection(collection_name)
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        return True


_shared_vector_store: QdrantVectorStoreAdapter | None = None


def get_vector_store() -> QdrantVectorStoreAdapter:
    """Get or create the singleton Qdrant vector store adapter."""
    global _shared_vector_store
    if _shared_vector_store is None:
        _shared_vector_store = QdrantVectorStoreAdapter()
    return _shared_vector_store

