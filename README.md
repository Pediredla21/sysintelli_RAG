# Legal Document RAG Chatbot MVP

This is a production-ready MVP of a Legal Document RAG (Retrieval-Augmented Generation) Chatbot. It allows users to upload legal PDF documents, processes them to extract and chunk text, and stores their embeddings in a local vector database. Users can then ask questions based on these documents, and an LLM will generate answers strictly relying on the provided context, citing the retrieved chunks.

## Architecture

The project employs a robust local technology stack prioritizing modularity, reusability, and offline capabilities:

- **Backend:** FastAPI for handling API requests (`/upload` and `/ask`).
- **Text Extraction & Chunking:** LangChain `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) via LangChain.
- **Vector Database:** Local FAISS index for efficiently storing and retrieving high-dimensional chunk embeddings.
- **LLM Engine:** Local Ollama instance (configurable, default: `mistral`) executing LangChain pipeline prompts.
- **Frontend:** Streamlit web app providing a simple chat interface and file uploader.

## Prerequisites

1. **Python 3.10+**
2. **Ollama:** Must be installed on your local machine.
   - Download from [Ollama.com](https://ollama.com/)
   - Pull the specific model you'd like to use. By default, the application is configured to use `mistral`. Run the following command in your terminal:
     ```bash
     ollama pull mistral
     ```

## Setup Instructions

1. **Clone the repository and navigate inside the folder**
   *(Assuming you are already in the project directory)*

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Backend (FastAPI)

The backend handles the core RAG lifecycle: PDF extraction, vector database indexing, and LLM inference.

1. Open a new terminal instance and activate the virtual environment.
2. Ensure you are in the root directory (`legal-rag-chatbot/`).
3. Start the FastAPI server using `uvicorn`:
   ```bash
   uvicorn app.main:app --reload
   ```
4. The API will run locally at `http://127.0.0.1:8000`. You can test or view the endpoint documentation at `http://127.0.0.1:8000/docs`.

## Running the Frontend (Streamlit)

The frontend provides the UI for interacting with the backend.

1. Open another new terminal instance and activate the virtual environment.
2. Ensure you are in the root directory.
3. Start the Streamlit application:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
4. A browser window should automatically open pointing to `http://localhost:8501`.

## How the RAG Pipeline Works

1. **Upload Phase:**
   - The user uploads a PDF via the Streamlit interface.
   - The file is sent to the FastAPI `POST /upload` endpoint.
   - `PyPDFLoader` extracts text from the document.
   - `RecursiveCharacterTextSplitter` chunks the content into segments of 1000 characters with an overlap of 200.
   - `HuggingFaceEmbeddings` converts these chunks into dense vectors.
   - A FAISS vector database is generated from these vectors and saved locally to the `data/` directory.

2. **Retrieve & Answer Phase:**
   - The user inputs a question in the chat interface.
   - The question is sent to the FastAPI `POST /ask` endpoint.
   - A similarity search retrieves the top 3 most relevant chunks from the FAISS vector database.
   - A custom prompt template concatenates the retrieved context chunks and states that the prompt must be strictly answered using *only* the provided context (or return "Information not found in document" otherwise).
   - Ollama generates an answer from the prompt.
   - The response is returned to the frontend along with the source chunks to ensure full traceability.
