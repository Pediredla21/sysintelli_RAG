import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import get_embedding_model
from .config import settings
import logging

logger = logging.getLogger(__name__)

def save_vector_store(documents: List[Document]) -> bool:
    """
    Creates a new FAISS vector store from text chunks/documents and saves it locally.
    Overwrites the previous index if it exists.
    """
    embeddings = get_embedding_model()
    try:
        logger.info(f"Creating FAISS vector index from {len(documents)} chunks...")
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save to local disk
        logger.info(f"Saving vector index to {settings.FAISS_INDEX_DIR}...")
        vector_store.save_local(settings.FAISS_INDEX_DIR)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create or save vector store: {str(e)}")
        raise e

def load_vector_store() -> FAISS | None:
    """
    Loads the FAISS vector store from the local directory if it exists.
    Returns None if no index is found.
    """
    embeddings = get_embedding_model()
    index_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    
    if not os.path.exists(index_path):
        logger.warning(f"No FAISS index found at {settings.FAISS_INDEX_DIR}")
        return None
        
    try:
        # Load local index; allow_dangerous_deserialization is required for local FAISS loading
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_DIR, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        return None
