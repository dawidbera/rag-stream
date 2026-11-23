"""Ingestion service coordinating parsing, hierarchical chunking, and vector indexing."""

from typing import Any

from rag_stream.adapters.embeddings import BM25SparseTokenizer, DenseEmbeddingAdapter
from rag_stream.adapters.vector_store import QdrantVectorStoreAdapter
from rag_stream.config import settings
from rag_stream.domain.models import Document, DocumentChunk
from rag_stream.services.chunker import HierarchicalChunker
from rag_stream.services.parser import DocumentParser


class IngestionService:
    """Manages full document ingestion pipeline into Qdrant."""

    def __init__(
        self,
        vector_store: QdrantVectorStoreAdapter | None = None,
        dense_embedder: DenseEmbeddingAdapter | None = None,
        sparse_tokenizer: BM25SparseTokenizer | None = None,
        chunker: HierarchicalChunker | None = None,
    ):
        """Initialize pipeline adapters and chunker."""
        self.vector_store = vector_store or QdrantVectorStoreAdapter()
        self.dense_embedder = dense_embedder or DenseEmbeddingAdapter()
        self.sparse_tokenizer = sparse_tokenizer or BM25SparseTokenizer()
        self.chunker = chunker or HierarchicalChunker()

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str = settings.default_collection,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[DocumentChunk]]:
        """Parse raw document, generate hierarchical chunks, compute embeddings, and index.

        Args:
            file_bytes: Raw file bytes.
            filename: Original filename.
            collection_name: Target Qdrant collection name.
            metadata: Optional additional metadata.

        Returns:
            Tuple of (Document, list of created DocumentChunks).
        """
        doc = Document(
            filename=filename,
            raw_text="",
            metadata=metadata or {},
        )

        parsed_doc = DocumentParser.parse(
            file_bytes=file_bytes,
            filename=filename,
            document_id=doc.document_id,
            custom_metadata=metadata,
        )
        doc.raw_text = parsed_doc.full_text

        parent_chunks, child_chunks = self.chunker.chunk_document(parsed_doc)

        # Compute embeddings for child chunks
        child_texts = [c.content for c in child_chunks]
        dense_vectors = self.dense_embedder.embed_batch(child_texts)

        for chunk, dense_vec in zip(child_chunks, dense_vectors, strict=False):
            chunk.dense_vector = dense_vec
            chunk.sparse_vector = self.sparse_tokenizer.tokenize_sparse(chunk.content)

        # Index both parent and child chunks (parents have empty vectors for retrieval isolation)
        all_chunks = parent_chunks + child_chunks
        self.vector_store.upsert_chunks(
            collection_name=collection_name,
            chunks=all_chunks,
        )

        return doc, all_chunks

    def delete_document(
        self,
        document_id: str,
        collection_name: str = settings.default_collection,
    ) -> bool:
        """Remove document and all associated chunks from vector store."""
        return self.vector_store.delete_document(
            collection_name=collection_name,
            document_id=document_id,
        )
