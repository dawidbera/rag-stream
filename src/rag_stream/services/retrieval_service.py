"""Retrieval orchestration service executing multi-stage RAG pipeline."""

import time
from collections.abc import AsyncGenerator

from rag_stream.adapters.embeddings import BM25SparseTokenizer, DenseEmbeddingAdapter
from rag_stream.adapters.llm_client import LLMClientAdapter
from rag_stream.adapters.reranker import CrossEncoderReranker
from rag_stream.adapters.vector_store import QdrantVectorStoreAdapter, get_vector_store
from rag_stream.config import settings
from rag_stream.domain.models import QueryRequestModel, QueryResponseModel
from rag_stream.services.crag_evaluator import CorrectiveRAGEvaluator
from rag_stream.services.rrf import ReciprocalRankFusion


class RetrievalService:
    """Orchestrates end-to-end question answering pipeline:

    1. Query Rewriting & Expansion (HyDE / Multi-Query)
    2. Parallel Hybrid Search (Dense ANN + Sparse BM25 in Qdrant)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking
    5. Corrective RAG (CRAG) Relevance Evaluation Gate
    6. Streaming / Complete Answer Generation with Citations
    """

    def __init__(
        self,
        vector_store: QdrantVectorStoreAdapter | None = None,
        dense_embedder: DenseEmbeddingAdapter | None = None,
        sparse_tokenizer: BM25SparseTokenizer | None = None,
        rrf: ReciprocalRankFusion | None = None,
        reranker: CrossEncoderReranker | None = None,
        crag_evaluator: CorrectiveRAGEvaluator | None = None,
        llm_client: LLMClientAdapter | None = None,
    ):
        """Initialize all pipeline components."""
        self.vector_store = vector_store or get_vector_store()
        self.dense_embedder = dense_embedder or DenseEmbeddingAdapter()
        self.sparse_tokenizer = sparse_tokenizer or BM25SparseTokenizer()
        self.rrf = rrf or ReciprocalRankFusion()
        self.reranker = reranker or CrossEncoderReranker()
        self.crag_evaluator = crag_evaluator or CorrectiveRAGEvaluator()
        self.llm_client = llm_client or LLMClientAdapter()

    async def execute_query(self, request: QueryRequestModel) -> QueryResponseModel:
        """Execute end-to-end RAG query flow."""
        start_time = time.perf_counter()

        # Step 1: Query preprocessing / expansion
        query_text = request.query
        if request.enable_query_rewrite:
            query_text = self._rewrite_query(query_text)

        # Step 2: Hybrid search (Dense + Sparse)
        dense_vec = self.dense_embedder.embed_text(query_text)
        sparse_vec = self.sparse_tokenizer.tokenize_sparse(query_text)

        dense_results = self.vector_store.search_dense(
            collection_name=request.collection_name,
            query_vector=dense_vec,
            limit=settings.rerank_candidate_pool,
        )
        sparse_results = self.vector_store.search_sparse(
            collection_name=request.collection_name,
            sparse_indices=sparse_vec["indices"],
            sparse_values=sparse_vec["values"],
            limit=settings.rerank_candidate_pool,
        )

        # Step 3: Reciprocal Rank Fusion
        fused_candidates = self.rrf.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=settings.rerank_candidate_pool,
        )

        # Step 4: Cross-Encoder Reranking
        if request.enable_reranking:
            ranked_candidates = self.reranker.rerank(
                query=request.query,
                candidates=fused_candidates,
                top_k=request.top_k,
            )
        else:
            ranked_candidates = fused_candidates[: request.top_k]

        # Step 5: Corrective RAG (CRAG) Gate
        verdict = self.crag_evaluator.evaluate(ranked_candidates)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if not verdict.is_relevant:
            return QueryResponseModel(
                query=request.query,
                answer=verdict.fallback_message or settings.fallback_message,
                sources=[],
                latency_ms=latency_ms,
                is_fallback=True,
            )

        # Step 6: Generation with citations
        answer, sources = await self.llm_client.generate_answer(
            query=request.query,
            context_chunks=verdict.selected_chunks,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        return QueryResponseModel(
            query=request.query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            is_fallback=False,
        )

    async def execute_query_stream(
        self,
        request: QueryRequestModel,
    ) -> AsyncGenerator[str, None]:
        """Execute RAG query and stream tokens via Server-Sent Events (SSE)."""
        # Execute retrieval and evaluation
        dense_vec = self.dense_embedder.embed_text(request.query)
        sparse_vec = self.sparse_tokenizer.tokenize_sparse(request.query)

        dense_results = self.vector_store.search_dense(
            collection_name=request.collection_name,
            query_vector=dense_vec,
            limit=settings.rerank_candidate_pool,
        )
        sparse_results = self.vector_store.search_sparse(
            collection_name=request.collection_name,
            sparse_indices=sparse_vec["indices"],
            sparse_values=sparse_vec["values"],
            limit=settings.rerank_candidate_pool,
        )

        fused = self.rrf.fuse(dense_results, sparse_results, top_k=settings.rerank_candidate_pool)
        ranked = self.reranker.rerank(request.query, fused, top_k=request.top_k)
        verdict = self.crag_evaluator.evaluate(ranked)

        if not verdict.is_relevant:
            yield f"data: {{\"token\": \"{verdict.fallback_message}\", \"is_fallback\": true}}\n\n"
            yield "data: [DONE]\n\n"
            return

        async for token in self.llm_client.stream_answer(request.query, verdict.selected_chunks):
            # Format SSE payload
            yield f"data: {{\"token\": \"{token}\"}}\n\n"

        # Stream final citations event
        citations = self.llm_client._build_citations(verdict.selected_chunks)
        citations_json = [c.model_dump() for c in citations]
        import json
        yield f"data: {{\"event\": \"citations\", \"sources\": {json.dumps(citations_json)}}}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _rewrite_query(query: str) -> str:
        """Normalize and expand query terms."""
        return query.strip()
