"""Tests for DocumentParser."""

from rag_stream.services.parser import DocumentParser


def test_parse_markdown_preserves_tables_and_headings():
    """Verify markdown tables and headings are preserved properly."""
    md_content = """# Annual Report

| Metric | 2024 | 2025 |
| :--- | :--- | :--- |
| Revenue | $10M | $15M |
| Expenses | $4M | $6M |

Some conclusion text here.
"""
    parsed = DocumentParser.parse(
        file_bytes=md_content.encode("utf-8"),
        filename="report.md",
        document_id="doc_test_1",
    )
    assert parsed.document_id == "doc_test_1"
    assert parsed.filename == "report.md"
    assert "| Revenue | $10M | $15M |" in parsed.full_text
    assert "# Annual Report" in parsed.full_text
    assert len(parsed.pages) == 1


def test_parse_plain_text_collapses_whitespace():
    """Verify plain text normalizes excessive blank lines and spaces."""
    raw = "Line 1.\n\n\n\nLine 2   with   spaces.\n\nLine 3."
    parsed = DocumentParser.parse(
        file_bytes=raw.encode("utf-8"),
        filename="notes.txt",
        document_id="doc_test_2",
    )
    assert "Line 1." in parsed.full_text
    assert "Line 2 with spaces." in parsed.full_text
