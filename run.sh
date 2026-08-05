#!/bin/bash
set -e

echo "========================================"
echo "Starting MHT-CET Agentic RAG Counsellor"
echo "========================================"

# Create runtime directories
mkdir -p storage/chroma_db
mkdir -p storage/uploads
mkdir -p storage/output
mkdir -p sessions

echo "Storage directories created."

# Start FastAPI
echo "Starting FastAPI Backend..."

uvicorn api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info &

FASTAPI_PID=$!

echo "Waiting for FastAPI to start..."
sleep 10

# Check FastAPI
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "❌ FastAPI failed to start."
    exit 1
fi

echo "✅ FastAPI started successfully."

# Start Streamlit
echo "Starting Streamlit UI..."

exec streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=200 \
    --browser.gatherUsageStats=false