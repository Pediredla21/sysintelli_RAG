#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start FastAPI backend in the background
echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# Start Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
streamlit run frontend/streamlit_app.py --server.port 8501

# Trap Ctrl+C to kill the backend too
trap "kill $BACKEND_PID" EXIT
