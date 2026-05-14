from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi import Query

from app.rag_pipeline import ask_rag, index_documents

load_dotenv()

app = FastAPI()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str
    filename: str | None = None


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(request: AskRequest, debug: bool = Query(False)):
    result = ask_rag(request.question, debug=debug, filename=request.filename)
    return result


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_extensions = (".txt", ".md", ".pdf")
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are supported for now",
        )

    save_path = UPLOAD_DIR / file.filename

    content = await file.read()
    save_path.write_bytes(content)

    try:
        index_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return {
        "filename": file.filename,
        "saved_to": str(save_path),
        "message": "File uploaded and indexed successfully",
    }
