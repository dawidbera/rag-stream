"""Pytest test configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from rag_stream.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient instance for testing API endpoints."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate minimal valid PDF bytes for testing upload."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    import io
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def sample_markdown_bytes() -> bytes:
    """Generate sample Markdown document content."""
    md = """# Cloud Cost Financial Review 2025

## Infrastructure Spending
In Q3 2025, total cloud infrastructure expenses reached $4.2M, driven primarily by Kubernetes cluster scaling and GPU node reservations.

### Cost Breakdown
| Service | Cost |
| :--- | :--- |
| Kubernetes Clusters | $2.5M |
| Object Storage | $0.9M |
| Networking & Egress | $0.8M |
"""
    return md.encode("utf-8")
