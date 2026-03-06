from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging

from .document_loader import process_uploaded_pdf
from .vector_store import save_vector_store
from .rag_pipeline import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to handle PDF uploads, process them into chunks, and store the embeddings in a FAISS vector db.
    """
    logger.info(f"Received file for upload: {file.filename}")
    try:
        # 1. Process PDF into textual chunks
        chunks = await process_uploaded_pdf(file)
        
        # 2. Embed and save to vector store
        save_vector_store(chunks)
        
        return {
            "message": "File successfully uploaded and processed.",
            "chunks_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint to accept a question, search the DB, run the LLM, and return the answer.
    """
    logger.info(f"Received question: {request.question}")
    try:
        result = answer_question(request.question)
        
        # Serialize source documents safely
        sources = [
            {
                "content": getattr(doc, 'page_content', ''), 
                "metadata": getattr(doc, 'metadata', {})
            } 
            for doc in result.get("sources", [])
        ]
        
        return {
            "answer": result.get("answer", ""),
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
