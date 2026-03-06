import os
import tempfile
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import settings
import logging

logger = logging.getLogger(__name__)

async def process_uploaded_pdf(file: UploadFile):
    """
    Validates the uploaded file, saves it temporarily, extracts text using PyPDFLoader,
    and splits it into chunks using RecursiveCharacterTextSplitter.
    Raises HTTPException if there is an error validating or processing.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_file_path = ""
    try:
        # Save uploaded file temporarily to process with PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file_path = tmp.name
            
        logger.info(f"Processing temporary file: {temp_file_path}")
        
        # Load documents
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or unreadable.")
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Extracted {len(documents)} pages and split into {len(chunks)} chunks.")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
