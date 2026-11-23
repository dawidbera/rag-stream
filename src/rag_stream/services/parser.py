"""Document parser supporting PDF, Markdown, and tabular text normalization."""

import io
import re
from typing import Any

from pypdf import PdfReader

from rag_stream.domain.models import ParsedDocument


class DocumentParser:
    """Extracts, cleans, and structures text from raw files (PDF, Markdown, TXT).

    Preserves Markdown table syntax, normalizes erratic whitespaces, and
    maintains page-level mapping for document attribution.
    """

    @classmethod
    def parse(
        cls,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """Parse raw file bytes into a structured ParsedDocument.

        Args:
            file_bytes: Raw binary content of uploaded file.
            filename: Original filename.
            document_id: Unique ID for the document.
            custom_metadata: Optional extra metadata dictionary.

        Returns:
            Structured ParsedDocument with page-level segments and cleaned text.
        """
        metadata = custom_metadata or {}
        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
            return cls._parse_pdf(file_bytes, filename, document_id, metadata)
        elif lower_name.endswith((".md", ".markdown")):
            return cls._parse_markdown(file_bytes, filename, document_id, metadata)
        else:
            return cls._parse_plain_text(file_bytes, filename, document_id, metadata)

    @classmethod
    def _parse_pdf(
        cls,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> ParsedDocument:
        """Extract text from PDF pages using pypdf."""
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = cls.clean_text(extracted)
            if cleaned:
                pages.append({"page_number": idx, "content": cleaned})
                full_text_parts.append(cleaned)

        full_text = "\n\n".join(full_text_parts)
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            full_text=full_text,
            pages=pages,
            metadata=metadata,
        )

    @classmethod
    def _parse_markdown(
        cls,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> ParsedDocument:
        """Parse Markdown file, preserving headings and markdown tables."""
        raw_text = file_bytes.decode("utf-8", errors="replace")
        cleaned = cls.clean_text(raw_text, is_markdown=True)
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            full_text=cleaned,
            pages=[{"page_number": 1, "content": cleaned}],
            metadata=metadata,
        )

    @classmethod
    def _parse_plain_text(
        cls,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> ParsedDocument:
        """Parse plain text documents."""
        raw_text = file_bytes.decode("utf-8", errors="replace")
        cleaned = cls.clean_text(raw_text)
        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            full_text=cleaned,
            pages=[{"page_number": 1, "content": cleaned}],
            metadata=metadata,
        )

    @staticmethod
    def clean_text(text: str, is_markdown: bool = False) -> str:
        """Normalize line breaks and spaces while preserving markdown structures.

        Args:
            text: Input raw text.
            is_markdown: If True, preserves Markdown table pipes and list indentation.

        Returns:
            Cleaned and normalized text string.
        """
        # Replace carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if is_markdown:
            # Collapse more than two consecutive newlines into two
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Remove trailing whitespaces per line
            lines = [line.rstrip() for line in text.split("\n")]
            return "\n".join(lines).strip()

        # For standard text, collapse excessive whitespace while retaining paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        cleaned_paragraphs = [re.sub(r"\s+", " ", p) for p in paragraphs]
        return "\n\n".join(cleaned_paragraphs)
