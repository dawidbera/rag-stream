# Troubleshooting Guide

This guide details common operational issues, diagnostics, and recovery procedures for the RAG Stream service.

---

## 1. Vector Store Connection Issues

### Symptom: `ConnectionRefusedError: [Errno 111] Connection refused`
* **Root Cause:** Application cannot reach the Qdrant service on configured host/port (`QDRANT_URL=http://qdrant:6333`).
* **Diagnosis:**
  ```bash
  curl -I http://localhost:6333/readyz
  docker ps --filter "name=qdrant"
  ```
* **Resolution:**
  1. If running locally with Docker Compose, verify container status:
     ```bash
     docker compose ps
     ```
  2. If running standalone tests or development without Docker, unset `QDRANT_URL` in `.env` to fallback to zero-dependency in-memory mode:
     ```bash
     QDRANT_URL=
     ```

---

## 2. Low Relevance Fallbacks Triggered Prematurely

### Symptom: Query returns fallback message despite relevant documents being uploaded
* **Root Cause:** The Cross-Encoder or RRF relevance score falls below the configured Corrective RAG (CRAG) threshold (`CRAG_MIN_CONFIDENCE_THRESHOLD=0.35`).
* **Diagnosis:**
  * Check the query keywords against chunk vocabulary.
  * Ingested documents might use synonymous terms not matched during exact keyword or dense search.
* **Resolution:**
  * Adjust the threshold in `.env` or Helm `values.yaml`:
    ```bash
    CRAG_MIN_CONFIDENCE_THRESHOLD=0.25
    ```
  * Enable query expansion (`"enable_query_rewrite": true`) to expand short queries.

---

## 3. Server-Sent Events (SSE) Buffering in Reverse Proxies

### Symptom: Streaming endpoint `/api/v1/query/stream` pauses and delivers all tokens in a single batch
* **Root Cause:** Intermediate reverse proxies (such as NGINX, Cloudflare, or AWS ALB) buffer HTTP responses by default.
* **Resolution:**
  * In NGINX, disable proxy buffering for streaming routes:
    ```nginx
    location /api/v1/query/stream {
        proxy_pass http://rag_backend;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
    }
    ```

---

## 4. PDF Text Extraction Yields Empty Content

### Symptom: Upload returns `chunks_created: 0` or 400 Bad Request
* **Root Cause:** The uploaded PDF is a scanned raster image without an embedded digital text layer (OCR required).
* **Resolution:**
  * Ensure uploaded PDFs contain selectable text.
  * For scanned paper documents, pre-process documents with OCR tools (e.g. `tesseract` or `ocrmypdf`) prior to ingestion.
