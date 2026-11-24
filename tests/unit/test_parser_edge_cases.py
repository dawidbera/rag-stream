"""Edge case tests for parser."""

from rag_stream.services.parser import DocumentParser


def test_parser_empty_content():
    """Verify parser handles empty byte strings gracefully."""
    parsed = DocumentParser.parse(b"", "empty.txt", "doc_empty")
    assert parsed.full_text == ""
    assert len(parsed.pages) == 1


def test_parser_malformed_newlines():
    """Verify multiple trailing and carriage return variants are normalized."""
    text = "Heading\r\n\r\n\r\nParagraph with carriage returns.\r\n"
    parsed = DocumentParser.parse(text.encode("utf-8"), "test.md", "doc_cr")
    assert "\r" not in parsed.full_text
    assert "Heading" in parsed.full_text
