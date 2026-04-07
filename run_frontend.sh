#!/bin/bash
# Run frontend server

echo "🎨 Starting Toronto Bylaw Agent Frontend..."
echo "📍 Frontend will be available at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run frontend/app.py --server.port 8501
