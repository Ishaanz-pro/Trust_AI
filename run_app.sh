#!/bin/bash

# Trust AI - Application Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

set -e

echo "======================================"
echo "Trust AI - Ethical Loan Approval Platform"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Train model if it doesn't exist
if [ ! -f "models/loan_model.pkl" ]; then
    echo ""
    echo "Training initial model..."
    python3 src/models/loan_model.py
fi

# Start FastAPI backend in background
echo ""
echo "Starting FastAPI backend on http://localhost:8000..."
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
sleep 5

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Warning: API may not be fully ready yet..."
fi

# Start Streamlit frontend
echo ""
echo "Starting Streamlit UI on http://localhost:8501..."
echo ""
echo "======================================"
echo "Application is starting!"
echo "API: http://localhost:8000"
echo "UI: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo "======================================"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0

# Cleanup on exit
trap "echo 'Stopping services...'; kill $API_PID 2>/dev/null; exit" INT TERM

wait
