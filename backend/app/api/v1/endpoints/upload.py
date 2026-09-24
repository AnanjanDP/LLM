import os
import tempfile
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Depends, status, Query

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.exceptions import APIException
from app.schemas.document import DocumentOut, UploadResponse
from app.services.rag_service import rag_service
from app.services.auth_service import get_current_user_optional
from app.models.db_models import Document, User

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

router = APIRouter()

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/upload-doc", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: Optional[str] = Query("word", description="Chunking strategy ('word', 'sentence', 'paragraph')"),
    chunk_size: Optional[int] = Query(200, ge=20, le=2000, description="Chunk size"),
    chunk_overlap: Optional[int] = Query(40, ge=0, le=500, description="Chunk overlap"),
    session_id: Optional[str] = Query(None, description="Session ID to scope document embeddings"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Upload and index a document (.pdf, .txt, .md) with configurable chunking."""
    filename = file.filename or "uploaded_file.txt"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in (".txt", ".pdf", ".md"):
        raise APIException("Unsupported file type. Only .txt, .pdf, and .md files are allowed.", status_code=400)

    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise APIException("File exceeds maximum allowed size of 10 MB.", status_code=400)

    text_content = ""

    if extension in (".txt", ".md"):
        text_content = content.decode("utf-8", errors="ignore")

    elif extension == ".pdf":
        if PdfReader is None:
            raise APIException("PDF reader library not installed.", status_code=500)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            reader = PdfReader(tmp_path)
            extracted_pages = []
            for page in reader.pages:
                extracted_pages.append(page.extract_text() or "")
            text_content = "\n".join(extracted_pages)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not text_content.strip():
        raise APIException("Could not extract readable text from the uploaded document.", status_code=400)

    # Ingest document into vector store
    chunk_count = rag_service.add_document(
        text_content,
        doc_name=filename,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        session_id=session_id
    )

    # Persist document record in SQLite DB
    user_id = user.id if user else None
    doc_record = Document(
        user_id=user_id,
        filename=filename,
        file_path=f"data/{filename}",
        file_type=extension,
        file_size=file_size,
        chunk_count=chunk_count
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    return UploadResponse(
        message=f"Successfully indexed document '{filename}' with {chunk_count} chunks ({chunk_strategy} strategy).",
        document=DocumentOut.model_validate(doc_record)
    )


@router.delete("/clear-documents")
async def clear_documents(
    session_id: Optional[str] = Query(None, description="Clear documents for specific session or all if omitted")
):
    """Clear or reset persistent vector store collection."""
    rag_service.clear_index(session_id=session_id)
    return {"message": f"Successfully cleared vector store collection (session_id={session_id})."}

