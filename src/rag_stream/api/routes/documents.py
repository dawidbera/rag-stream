"""Document ingestion and management routes."""

import json
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rag_stream.api.schemas import DeleteDocumentResponse, DocumentUploadResponse
from rag_stream.config import settings
from rag_stream.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
ingestion_service = IngestionService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file (PDF, Markdown, or plain text)")],
    collection_name: Annotated[str, Form(description="Target Qdrant collection")] = settings.default_collection,
    metadata: Annotated[str | None, Form(description="Optional JSON-encoded metadata string")] = None,
) -> DocumentUploadResponse:
    """Upload and ingest a document into the Advanced RAG system."""
    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as err:
            raise HTTPException(status_code=400, detail=f"Invalid JSON metadata: {err}") from err

    filename = file.filename or "uploaded_document.txt"
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc, chunks = ingestion_service.ingest_document(
        file_bytes=file_bytes,
        filename=filename,
        collection_name=collection_name,
        metadata=parsed_metadata,
    )

    return DocumentUploadResponse(
        document_id=doc.document_id,
        filename=filename,
        chunks_created=len(chunks),
        status="INDEXED",
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    collection_name: str = settings.default_collection,
) -> DeleteDocumentResponse:
    """Delete a document and all corresponding chunks from the vector store."""
    success = ingestion_service.delete_document(
        document_id=document_id,
        collection_name=collection_name,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")

    return DeleteDocumentResponse(
        document_id=document_id,
        status="DELETED",
    )
