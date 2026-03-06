import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Legal Document RAG Chatbot MVP"
    DATA_DIR: str = "data"
    FAISS_INDEX_DIR: str = "data/faiss_index"
    
    # Text Splitting configurations
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # LLM Settings
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # RAG Settings
    RETRIEVER_K: int = 6
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instance to be imported
settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
