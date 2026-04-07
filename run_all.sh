#!/bin/bash
# Run both backend and frontend servers

echo "🏛️  Toronto Bylaw Agent - Starting All Services..."
echo ""
echo "This script will start both the backend and frontend servers."
echo "You can access the application at http://localhost:8501"
echo ""
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start backend in background
echo "Starting backend server..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend server..."
streamlit run frontend/app.py --server.port 8501

# Cleanup
kill $BACKEND_PID 2>/dev/null
