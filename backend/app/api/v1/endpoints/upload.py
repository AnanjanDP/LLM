import io
import os
import re
from typing import Literal, Optional
from urllib.parse import unquote
from fastapi import APIRouter, UploadFile, File, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import APIException
from app.core.limiter import limiter
from app.schemas.document import DocumentOut, UploadResponse
from app.services.rag_service import rag_service
from app.services.auth_service import get_current_user_optional
from app.services.conversation_service import conversation_service
from app.models.db_models import Document, User

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

router = APIRouter()

ALLOWED_EXTENSIONS = (".txt", ".pdf", ".md")


def _safe_filename(name: Optional[str]) -> str:
    """Keep only the base name, drop control characters, and cap the length (the name is stored and echoed back)."""
    # Browsers percent-encode control characters in multipart filenames, so decode before stripping them.
    base = os.path.basename(unquote(name or "").replace("\\", "/")) or "uploaded_file.txt"
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:100] or "uploaded_file.txt"


def _extract_pdf_text(content: bytes) -> str:
    if PdfReader is None:
        raise APIException("PDF reader library not installed.", status_code=500)
    try:
        reader = PdfReader(io.BytesIO(content))
        if len(reader.pages) > settings.MAX_PDF_PAGES:
            raise APIException(f"PDF has more than {settings.MAX_PDF_PAGES} pages.", status_code=400)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except APIException:
        raise
    except Exception:
        raise APIException("Could not read this PDF (it may be corrupt or encrypted).", status_code=400)


@router.post("/upload-doc", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
async def upload_document(
    request: Request,
    session_id: str = Query(..., min_length=1, max_length=64, description="Conversation the document belongs to"),
    file: UploadFile = File(...),
    chunk_strategy: Literal["word", "sentence", "paragraph"] = Query("word", description="Chunking strategy"),
    chunk_size: int = Query(200, ge=20, le=2000, description="Chunk size"),
    chunk_overlap: int = Query(40, ge=0, le=500, description="Chunk overlap"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Upload and index a document (.pdf, .txt, .md). It is searchable only from the conversation it was uploaded to."""
    user_id = user.id if user else None
    await conversation_service.assert_session_access(db, session_id, user_id)

    filename = _safe_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise APIException("Unsupported file type. Only .txt, .pdf, and .md files are allowed.", status_code=400)

    # Read one byte past the limit so an oversized upload is rejected without loading all of it into memory.
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise APIException(
            f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_MB} MB.", status_code=413
        )

    if extension == ".pdf":
        text_content = await run_in_threadpool(_extract_pdf_text, content)
    else:
        text_content = content.decode("utf-8", errors="replace")

    if not text_content.strip():
        raise APIException("Could not extract readable text from the uploaded document.", status_code=400)

    chunk_count = await run_in_threadpool(
        rag_service.add_document,
        text_content,
        doc_name=filename,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size - 1),
        session_id=session_id,
    )

    doc_record = Document(
        user_id=user_id,
        filename=filename,
        file_path=filename,
        file_type=extension,
        file_size=len(content),
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
    session_id: str = Query(..., min_length=1, max_length=64, description="Conversation whose documents to remove"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Remove the documents uploaded to one conversation. There is deliberately no "clear everything" route."""
    await conversation_service.assert_session_access(db, session_id, user.id if user else None)
    removed = await run_in_threadpool(rag_service.clear_index, session_id)
    return {"message": f"Removed {removed} indexed chunks for session '{session_id}'.", "removed": removed}
