#!/bin/bash

# TRUST-AI Complete Application Launcher
# This script starts both FastAPI backend and Streamlit frontend

set -e

PROJECT_DIR="/Users/shriram/Trust_AI"
cd "$PROJECT_DIR"

echo "🚀 ============================================"
echo "🚀 TRUST-AI: Responsible Decision Support"
echo "🚀 ============================================"
echo ""

# Kill any existing processes on ports 8000 and 8501
echo "🔧 Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
sleep 1

# Start FastAPI Backend
echo ""
echo "🔧 Starting FastAPI Backend on http://localhost:8000..."
python3 backendmain.py > /tmp/trust_ai_backend.log 2>&1 &
BACKEND_PID=$!
echo "   ✅ Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ Backend is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "🔧 Starting Streamlit Frontend on http://localhost:8501..."
python3 -m streamlit run streamlitapp.py --server.port 8501 --client.showErrorDetails=true > /tmp/trust_ai_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   ✅ Frontend PID: $FRONTEND_PID"

sleep 3

echo ""
echo "🎉 ============================================"
echo "🎉 TRUST-AI Application is Ready!"
echo "🎉 ============================================"
echo ""
echo "📊 Frontend: http://localhost:8501"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "✨ Application Features:"
echo "   • Loan Decision Prediction with Explainability"
echo "   • SHAP-style Feature Importance Visualization"
echo "   • Fairness Audit & Metrics"
echo "   • Real-time ML Model Inference"
echo ""
echo "🔍 View logs:"
echo "   Backend: tail -f /tmp/trust_ai_backend.log"
echo "   Frontend: tail -f /tmp/trust_ai_frontend.log"
echo ""
echo "⚠️  To stop the application, run: kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Keep the script running and show logs
echo "📡 Application running..."
wait
