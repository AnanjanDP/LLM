from fastapi import APIRouter, UploadFile, File
from app.services.rag import add_document
from PyPDF2 import PdfReader
import tempfile

router = APIRouter()


@router.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        # ✅ TXT
        if file.filename.endswith(".txt"):
            content = await file.read()
            text = content.decode("utf-8")

        # ✅ PDF
        elif file.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            reader = PdfReader(tmp_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

        else:
            return {"error": "Only .txt and .pdf supported"}

        add_document(text)

        return {"message": "Document added successfully"}

    except Exception as e:
        return {"error": str(e)}