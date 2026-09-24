from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    message: str
    document: DocumentOut
