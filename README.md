# Toronto City Bylaw Conversational Agent

A Python-based intelligent agent for Toronto municipal services, built with FastAPI backend and Streamlit frontend.

## 🎯 Features

### Core Capabilities
- **Bylaw Guidance**: Answer questions about Toronto municipal bylaws and regulations
- **Hazard Reporting**: Multi-turn dialogue to collect and submit hazard reports to Toronto 311
- **Permit Screening**: Determine if a building permit is required for a project
- **Waste Collection Lookup**: Find waste collection schedules by postal code
- **Conversation Memory**: Maintain context across multi-turn conversations
- **RAG (Retrieval-Augmented Generation)**: Search knowledge base for accurate information
- **Safety Guardrails**: Detect and reject out-of-scope requests

### Technical Stack
- **Backend**: FastAPI (Python web framework)
- **Frontend**: Streamlit (Python web UI framework)
- **Database**: SQLite (local data storage)
- **RAG**: LangChain + ChromaDB (semantic search)
- **LLM**: Manus Forge API (qwen3-30b-a3b-fp8 model)

## 📋 Project Structure

```
toronto-bylaw-agent-python/
├── backend/
│   ├── config.py          # Configuration management
│   ├── database.py        # SQLAlchemy database setup
│   ├── models.py          # Database models
│   ├── rag.py             # RAG knowledge base
│   ├── llm.py             # LLM integration
│   ├── agent.py           # Agent logic (intent classification, routing, actions)
│   └── main.py            # FastAPI application
├── frontend/
│   └── app.py             # Streamlit UI
├── tests/
│   └── test_agent.py      # Test suite
├── data/
│   └── knowledge_base.json # Toronto bylaw documents
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
export LLM_API_URL=https://forge.manus.im/v1/chat/completions
export LLM_API_KEY=your_api_key_here
export LLM_MODEL=qwen3-30b-a3b-fp8
export DATABASE_URL=sqlite:///./toronto_bylaw.db
export BACKEND_HOST=0.0.0.0
export BACKEND_PORT=8000
export FRONTEND_PORT=8501
```

### 3. Initialize Database

```bash
python3 -c "from backend.database import init_db; init_db()"
```

### 4. Start Backend Server

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start Frontend (in another terminal)

```bash
streamlit run frontend/app.py --server.port 8501
```

### 6. Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 🧠 Agent Architecture

### Intent Classification
The agent classifies user messages into 5 categories:
1. **General Inquiry**: Questions about bylaws and regulations
2. **Hazard Report**: Reports of potholes, fallen trees, debris, etc.
3. **Permit Screener**: Determining if a permit is needed
4. **Collection Lookup**: Waste collection schedule queries
5. **Out-of-Scope**: Requests outside the agent's domain

### Multi-Turn Actions

#### Hazard Reporter
Collects hazard information through multi-turn dialogue:
1. Ask for location
2. Ask for hazard type
3. Ask for description
4. Submit ticket to Toronto 311 (mock)

#### Permit Screener
Analyzes project description to determine permit requirements:
- New construction → Permit required
- Major renovations (>25% property value) → Permit required
- Structural changes → Permit required
- Minor repairs → No permit needed

#### Collection Lookup
Extracts postal code from user message and provides:
- Collection day based on postal code
- Collection time window
- Bin types and guidelines

### RAG Knowledge Base
- **Storage**: ChromaDB for vector embeddings
- **Documents**: Toronto bylaws, regulations, service information
- **Search**: Semantic search using embeddings
- **Citations**: Provides sources for all responses

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test coverage includes:
- Intent classification accuracy
- Multi-turn dialogue flows
- Action execution
- Out-of-scope detection
- RAG retrieval quality

## 📊 API Endpoints

### Conversations
- `POST /conversations` - Create new conversation
- `GET /conversations` - List all conversations
- `GET /conversations/{id}/messages` - Get conversation messages

### Messages
- `POST /conversations/{id}/messages` - Send message and get agent response
- `GET /conversations/{id}/messages` - Get all messages in conversation

### Health
- `GET /health` - Health check

## 🔧 Configuration

All configuration is managed through environment variables in `backend/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_API_URL | forge.manus.im | LLM API endpoint |
| LLM_API_KEY | - | LLM API key |
| LLM_MODEL | qwen3-30b-a3b-fp8 | LLM model name |
| DATABASE_URL | sqlite:///./toronto_bylaw.db | Database connection string |
| BACKEND_HOST | 0.0.0.0 | Backend server host |
| BACKEND_PORT | 8000 | Backend server port |
| FRONTEND_PORT | 8501 | Frontend server port |
| CHROMA_DB_PATH | ./data/chroma_db | ChromaDB storage path |
| KNOWLEDGE_BASE_PATH | ./data/knowledge_base.json | Knowledge base file path |

## 📝 Knowledge Base

The knowledge base is stored in `data/knowledge_base.json` and includes:
- Toronto Municipal Code chapters
- Building permit requirements
- Waste collection information
- Hazard reporting procedures
- Property standards

To add more documents:
1. Edit `data/knowledge_base.json`
2. Restart the backend
3. Knowledge base will be automatically re-indexed

## 🛡️ Safety Features

### Guardrails
- **Out-of-Scope Detection**: Rejects requests not related to Toronto municipal services
- **Input Validation**: Validates user inputs before processing
- **Error Handling**: Graceful error handling with informative messages
- **Rate Limiting**: Can be added via FastAPI middleware

### Privacy
- **Local Storage**: All data stored locally in SQLite
- **No Data Sharing**: No data sent to external services except LLM API
- **Session Management**: Conversations isolated per user

## 🚢 Deployment

### Local Deployment
```bash
# Terminal 1: Backend
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/app.py --server.port 8501
```

### Docker Deployment (Optional)
Create a `Dockerfile` for containerization and deploy to cloud platforms like:
- Heroku
- AWS EC2
- Google Cloud Run
- Azure Container Instances

## 📚 Learning Resources

This project demonstrates:
- **FastAPI**: Modern Python web framework with automatic API documentation
- **Streamlit**: Rapid prototyping of data applications
- **SQLAlchemy**: ORM for database operations
- **LangChain**: LLM orchestration and RAG patterns
- **ChromaDB**: Vector database for semantic search
- **Async Programming**: Asynchronous request handling in Python

## 🤝 Contributing

To extend the agent:
1. Add new documents to `data/knowledge_base.json`
2. Implement new action handlers in `backend/agent.py`
3. Add tests in `tests/test_agent.py`
4. Update frontend in `frontend/app.py`

## 📞 Support

For issues or questions:
- Check the test suite for examples
- Review API documentation at `http://localhost:8000/docs`
- Contact Toronto 311 for municipal service questions

## 📄 License

This project is created for educational purposes.

---

**Created for**: Python Course Final Project
**Language**: Python 3.11+
**Last Updated**: 2026-04-07
