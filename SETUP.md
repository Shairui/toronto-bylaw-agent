# Setup Instructions for Toronto Bylaw Agent

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation Steps

### 1. Clone or Download the Project

```bash
cd /path/to/toronto-bylaw-agent-python
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root or export variables:

```bash
export LLM_API_URL=https://forge.manus.im/v1/chat/completions
export LLM_API_KEY=your_manus_api_key
export LLM_MODEL=qwen3-30b-a3b-fp8
export DATABASE_URL=sqlite:///./toronto_bylaw.db
export BACKEND_HOST=0.0.0.0
export BACKEND_PORT=8000
export FRONTEND_PORT=8501
```

### 5. Initialize Database

```bash
python3 -c "from backend.database import init_db; init_db()"
```

This creates the SQLite database and all necessary tables.

### 6. Initialize RAG Knowledge Base

The knowledge base is automatically initialized when the backend starts. It loads documents from `data/knowledge_base.json`.

## Running the Application

### Option A: Run Both Services (Recommended)

```bash
./run_all.sh
```

This will start:
- Backend API on http://localhost:8000
- Frontend UI on http://localhost:8501

### Option B: Run Services Separately

**Terminal 1 - Backend:**
```bash
./run_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./run_frontend.sh
```

### Option C: Manual Startup

**Backend:**
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
streamlit run frontend/app.py --server.port 8501
```

## Accessing the Application

Once both services are running:

1. **Frontend UI**: Open http://localhost:8501 in your browser
2. **Backend API**: http://localhost:8000
3. **API Documentation**: http://localhost:8000/docs (Swagger UI)
4. **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

## Troubleshooting

### Port Already in Use

If port 8000 or 8501 is already in use:

```bash
# Find process using port 8000
lsof -i :8000

# Find process using port 8501
lsof -i :8501

# Kill process (replace PID with actual process ID)
kill -9 PID
```

Or change ports in `backend/config.py`:
```python
BACKEND_PORT = 8001  # Change from 8000
FRONTEND_PORT = 8502  # Change from 8501
```

### Missing Dependencies

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

### Database Issues

To reset the database:

```bash
rm toronto_bylaw.db
python3 -c "from backend.database import init_db; init_db()"
```

### LLM API Errors

Ensure your LLM_API_KEY is correct:

```bash
echo $LLM_API_KEY
```

Test the API connection:
```bash
python3 -c "from backend.llm import llm_client; print('LLM client initialized')"
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_agent.py::test_intent_classification_general_inquiry -v
```

## Project Structure

```
toronto-bylaw-agent-python/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── database.py        # Database setup
│   ├── models.py          # SQLAlchemy models
│   ├── rag.py             # RAG knowledge base
│   ├── llm.py             # LLM integration
│   ├── agent.py           # Agent logic
│   └── main.py            # FastAPI app
├── frontend/
│   ├── __init__.py
│   └── app.py             # Streamlit UI
├── tests/
│   ├── __init__.py
│   └── test_agent.py      # Test suite
├── data/
│   └── knowledge_base.json # Toronto bylaws
├── requirements.txt        # Dependencies
├── run_all.sh             # Start all services
├── run_backend.sh         # Start backend only
├── run_frontend.sh        # Start frontend only
├── README.md              # Project documentation
└── SETUP.md               # This file
```

## Next Steps

1. **Customize Knowledge Base**: Edit `data/knowledge_base.json` to add more Toronto bylaw documents
2. **Extend Agent**: Add new action handlers in `backend/agent.py`
3. **Improve UI**: Customize `frontend/app.py` styling and layout
4. **Deploy**: Use Docker or cloud platforms for deployment

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review test cases in `tests/test_agent.py` for usage examples
- Check API documentation at http://localhost:8000/docs

---

**Happy coding! 🎉**
