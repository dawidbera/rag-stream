"""Latency and retrieval performance benchmark tests."""

import time

from fastapi.testclient import TestClient


def test_retrieval_latency_benchmark(client: TestClient, sample_markdown_bytes: bytes):
    """Verify that retrieval and evaluation pipeline executes within low-latency targets (<350ms)."""
    # 1. Ingest document
    files = {
        "file": ("benchmark_doc.md", sample_markdown_bytes, "text/markdown"),
    }
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"collection_name": "benchmark_col"},
    )

    # 2. Measure multiple query runs
    latencies: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = client.post(
            "/api/v1/query",
            json={
                "query": "Kubernetes cluster scaling expenses",
                "collection_name": "benchmark_col",
                "top_k": 3,
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        assert res.status_code == 200

    avg_latency = sum(latencies) / len(latencies)
    # Average latency should easily be under 350ms
    assert avg_latency < 350.0
