"""Integration tests for document upload and deletion endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoints(client: TestClient):
    """Test health check status endpoints."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "components" in data


def test_upload_markdown_document(client: TestClient, sample_markdown_bytes: bytes):
    """Test uploading a Markdown document via multipart/form-data."""
    files = {
        "file": ("financial_review.md", sample_markdown_bytes, "text/markdown"),
    }
    data = {
        "collection_name": "test_docs",
        "metadata": '{"author": "Financial Analysis Team", "year": 2025}',
    }
    response = client.post("/api/v1/documents/upload", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "INDEXED"
    assert res["filename"] == "financial_review.md"
    assert res["chunks_created"] > 0
    assert "doc_" in res["document_id"]


def test_delete_document_endpoint(client: TestClient, sample_markdown_bytes: bytes):
    """Test deleting an uploaded document."""
    files = {
        "file": ("to_delete.md", sample_markdown_bytes, "text/markdown"),
    }
    upload_res = client.post("/api/v1/documents/upload", files=files)
    doc_id = upload_res.json()["document_id"]

    del_res = client.delete(f"/api/v1/documents/{doc_id}?collection_name=documents")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "DELETED"
