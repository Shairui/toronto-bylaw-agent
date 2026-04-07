#!/bin/bash
# Run backend server

echo "🚀 Starting Toronto Bylaw Agent Backend..."
echo "📍 Backend will be available at http://localhost:8000"
echo "📚 API documentation at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
