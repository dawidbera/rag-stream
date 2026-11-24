"""Integration tests for query and streaming endpoints."""

from fastapi.testclient import TestClient


def test_query_documents_end_to_end(client: TestClient, sample_markdown_bytes: bytes):
    """Upload document and query answer with citations."""
    files = {
        "file": ("annual_report_2025.md", sample_markdown_bytes, "text/markdown"),
    }
    upload_res = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"collection_name": "finance_reports"},
    )
    assert upload_res.status_code == 200

    payload = {
        "query": "What were the total cloud infrastructure expenses in Q3?",
        "collection_name": "finance_reports",
        "top_k": 5,
        "enable_reranking": True,
        "stream": False,
    }
    query_res = client.post("/api/v1/query", json=payload)
    assert query_res.status_code == 200

    data = query_res.json()
    assert data["query"] == payload["query"]
    assert len(data["sources"]) > 0
    assert "financial_review" in data["sources"][0]["filename"] or "annual_report" in data["sources"][0]["filename"]
    assert data["latency_ms"] >= 0


def test_query_documents_stream(client: TestClient, sample_markdown_bytes: bytes):
    """Test SSE streaming response endpoint."""
    files = {
        "file": ("stream_doc.md", sample_markdown_bytes, "text/markdown"),
    }
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"collection_name": "stream_test"},
    )

    payload = {
        "query": "Kubernetes cluster scaling",
        "collection_name": "stream_test",
        "top_k": 3,
    }
    response = client.post("/api/v1/query/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data:" in body
    assert "[DONE]" in body
