import logging
from fastapi import FastAPI
from .routes import router

# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Legal Document RAG Chatbot MVP",
    description="A FastAPI backend for a Retrieval-Augmented Generation chatbot specialized in legal documents.",
    version="1.0.0"
)

# Register endpoints
app.include_router(router)

@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {"message": "Legal Document RAG API is running."}
