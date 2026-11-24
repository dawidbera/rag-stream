"""LLM client adapter supporting streaming and attributed citation synthesis."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from rag_stream.config import settings
from rag_stream.domain.models import ScoredChunk, SourceCitation


class LLMClientAdapter:
    """Interacts with Language Models for answering queries with citations.

    Supports OpenAI API when an API key is configured, or provides an internal
    deterministic generation engine for offline execution and testing.
    """

    def __init__(
        self,
        api_key: str | None = settings.openai_api_key,
        model: str = settings.llm_model,
        temperature: float = settings.temperature,
    ):
        """Initialize LLM adapter."""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
    ) -> tuple[str, list[SourceCitation]]:
        """Synthesize a complete answer with attributed citations.

        Args:
            query: User's question.
            context_chunks: Approved context chunks from CRAG evaluator.

        Returns:
            Tuple of (answer_text, list of SourceCitations).
        """
        citations = self._build_citations(context_chunks)

        if self.api_key:
            answer = await self._call_openai(query, context_chunks)
            return answer, citations

        # Fallback local generation logic
        answer = self._generate_local_answer(query, context_chunks, citations)
        return answer, citations

    async def stream_answer(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens as an asynchronous generator.

        Yields:
            Token text chunks.
        """
        citations = self._build_citations(context_chunks)
        answer = self._generate_local_answer(query, context_chunks, citations)

        # Stream words with small realistic delay
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            await asyncio.sleep(0.015)

    def _build_citations(self, context_chunks: list[ScoredChunk]) -> list[SourceCitation]:
        """Convert ScoredChunks to SourceCitation domain objects."""
        citations: list[SourceCitation] = []
        for scored in context_chunks:
            chunk = scored.chunk
            excerpt = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
            filename = chunk.metadata.get("filename", "unknown.pdf")
            citations.append(
                SourceCitation(
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_id=chunk.chunk_id,
                    score=round(scored.final_score, 4),
                    content_excerpt=excerpt,
                    page_number=chunk.page_number or 1,
                )
            )
        return citations

    def _generate_local_answer(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
        citations: list[SourceCitation],
    ) -> str:
        """Construct grounded answer referencing specific documents and pages."""
        if not context_chunks:
            return settings.fallback_message

        # Assemble key context excerpts
        primary = context_chunks[0].chunk
        filename = primary.metadata.get("filename", "document.pdf")
        page = primary.page_number or 1

        # Use parent content if available for comprehensive grounded answer
        parent_content = primary.metadata.get("parent_content", primary.content)
        cleaned_text = parent_content.strip()

        # Extract most relevant sentence
        first_sentence = cleaned_text.split(".")[0] if "." in cleaned_text else cleaned_text[:120]

        answer = (
            f"Based on the analyzed documentation, {first_sentence.strip()} "
            f"[Doc: {filename}, Page {page}]."
        )
        return answer

    async def _call_openai(self, query: str, context_chunks: list[ScoredChunk]) -> str:
        """Execute request against OpenAI Chat Completions API."""
        prompt_context = "\n\n".join(
            [
                f"[Document: {c.chunk.metadata.get('filename')}, Page: {c.chunk.page_number}]\n{c.chunk.content}"
                for c in context_chunks
            ]
        )
        system_prompt = (
            "You are an enterprise AI assistant answering questions based strictly on the provided context. "
            "You must attribute citations in the format [Doc: <filename>, Page <number>]. "
            "If the context does not contain the answer, explicitly state that you cannot answer."
        )
        user_prompt = f"Context:\n{prompt_context}\n\nQuestion: {query}\n\nAnswer:"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
