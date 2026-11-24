"""Integration tests for PDF ingestion and validation."""

from fastapi.testclient import TestClient


def test_upload_pdf_document(client: TestClient, sample_pdf_bytes: bytes):
    """Test uploading a binary PDF document."""
    files = {
        "file": ("sample_report.pdf", sample_pdf_bytes, "application/pdf"),
    }
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "INDEXED"
    assert res["filename"] == "sample_report.pdf"


def test_upload_empty_file_fails(client: TestClient):
    """Test uploading empty bytes returns 400 error."""
    files = {
        "file": ("empty.txt", b"", "text/plain"),
    }
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_invalid_json_metadata_fails(client: TestClient, sample_markdown_bytes: bytes):
    """Test uploading malformed JSON in metadata returns 400 error."""
    files = {
        "file": ("valid.md", sample_markdown_bytes, "text/markdown"),
    }
    data = {
        "metadata": "{invalid_json: 123",
    }
    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]
