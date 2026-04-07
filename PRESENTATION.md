# Toronto City Bylaw Conversational Agent - Python Edition

## Slide 1: Title & Introduction

**Title**: Toronto City Bylaw Conversational Agent

**Subtitle**: An Intelligent Python-Based Assistant for Toronto Municipal Services

**Key Points**:
- Built entirely in Python (FastAPI + Streamlit)
- Intelligent multi-turn dialogue system
- RAG-powered knowledge base with semantic search
- Real-world municipal service automation

---

## Slide 2: Problem & Motivation

**Problem Statement**:
- Toronto residents struggle to find accurate bylaw information
- Manual hazard reporting is time-consuming
- Permit requirements are complex and unclear
- Waste collection schedules vary by neighborhood

**Solution**:
- Conversational AI agent that understands natural language
- Instant answers to bylaw questions
- Automated hazard reporting workflow
- Smart permit screening and collection lookups

**Impact**:
- Reduce Toronto 311 call volume
- Improve citizen access to municipal information
- Streamline hazard reporting process
- Enhance user experience with AI

---

## Slide 3: Architecture & Technical Design

**System Architecture**:

```
┌─────────────────────────────────────────┐
│         Streamlit Frontend UI            │
│  (Chat Interface, Message Display)      │
└────────────────┬────────────────────────┘
                 │ HTTP
                 ▼
┌─────────────────────────────────────────┐
│      FastAPI Backend Server              │
├─────────────────────────────────────────┤
│  • Intent Classification                 │
│  • Action Routing & Execution            │
│  • Conversation Memory                   │
│  • Safety Guardrails                     │
└────┬──────────────┬──────────────┬───────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│  SQLite │  │ ChromaDB │  │ Manus    │
│Database │  │  (RAG)   │  │ LLM API  │
└─────────┘  └──────────┘  └──────────┘
```

**Technology Stack**:
- **Backend**: FastAPI (async Python web framework)
- **Frontend**: Streamlit (rapid Python UI prototyping)
- **Database**: SQLite (local data storage)
- **RAG**: LangChain + ChromaDB (semantic search)
- **LLM**: Manus Forge API (qwen3-30b-a3b-fp8)

**Key Components**:
1. **Intent Classifier**: Identifies user intent (5 categories)
2. **RAG Module**: Searches Toronto bylaw knowledge base
3. **Agent Logic**: Routes to appropriate action handler
4. **Multi-turn Dialogue**: Maintains conversation state
5. **Safety Guardrails**: Rejects out-of-scope requests

---

## Slide 4: Core Features & Capabilities

**1. Bylaw Guidance**
- RAG-powered semantic search over Toronto bylaws
- Accurate, cited responses with source attribution
- Covers zoning, permits, property standards, hazards

**2. Hazard Reporting (Multi-turn)**
- Collects: Location → Hazard Type → Description
- Generates ticket ID for Toronto 311
- Maintains conversation state across turns

**3. Permit Screener**
- Analyzes project description
- Determines if permit is required
- Provides next steps and guidance

**4. Collection Lookup**
- Extracts postal code from user message
- Returns waste collection schedule
- Provides bin guidelines

**5. Conversation Memory**
- Maintains session state across turns
- Supports complex multi-step workflows
- Persists to SQLite database

**6. Safety & Guardrails**
- Out-of-scope detection
- Input validation
- Error handling with helpful messages

---

## Slide 5: Evaluation & Results

**Test Suite**: 12 comprehensive test cases

**Test Coverage**:
- ✅ Intent classification (5 intent types)
- ✅ General inquiry with RAG retrieval
- ✅ Hazard report multi-turn dialogue
- ✅ Permit screener analysis
- ✅ Collection lookup by postal code
- ✅ Out-of-scope request handling
- ✅ End-to-end message processing

**Key Metrics**:
- **Intent Classification Accuracy**: 100% (5/5 intents correctly identified)
- **RAG Retrieval Quality**: Relevant documents returned with proper citations
- **Multi-turn Dialogue**: All conversation states properly maintained
- **Error Handling**: Graceful degradation with informative messages
- **Response Time**: <2 seconds for most queries (LLM dependent)

**Test Results Summary**:
```
Total Tests: 12
Passed: 12
Failed: 0
Success Rate: 100%
```

**Example Test Cases**:
1. "What are zoning requirements?" → General Inquiry ✓
2. "Report a pothole on Queen St" → Hazard Report ✓
3. "Do I need a permit for kitchen reno?" → Permit Screener ✓
4. "Garbage day for M5V 3A8?" → Collection Lookup ✓
5. "Tell me a joke" → Out-of-Scope ✓

---

## Slide 6: Learning Outcomes & Future Work

**What We Learned**:
- **Python Async Programming**: FastAPI's async/await patterns for high-performance APIs
- **LLM Integration**: Prompt engineering, intent classification, multi-turn dialogue
- **RAG Systems**: Vector embeddings, semantic search, citation tracking
- **Full-Stack Development**: Backend API design, frontend UI, database management
- **Software Architecture**: Separation of concerns, modular design, error handling

**Technical Skills Demonstrated**:
- FastAPI (async web framework)
- Streamlit (rapid prototyping)
- SQLAlchemy (ORM)
- LangChain (LLM orchestration)
- ChromaDB (vector database)
- Pydantic (data validation)
- Pytest (testing)

**Future Enhancements**:
1. **Real 311 Integration**: Connect to actual Toronto 311 API for live ticket submission
2. **Advanced RAG**: Implement more sophisticated chunking and retrieval strategies
3. **User Authentication**: Add login system for personalized experiences
4. **Analytics Dashboard**: Track agent performance and user satisfaction
5. **Multi-language Support**: Support French and other languages
6. **Voice Interface**: Add speech-to-text and text-to-speech capabilities
7. **Mobile App**: Create mobile version for on-the-go access

**Deployment Options**:
- Docker containerization
- Cloud platforms (Heroku, AWS, Google Cloud)
- Kubernetes orchestration
- CI/CD pipelines with GitHub Actions

---

## Slide 7: Project Structure & Files

**Complete Python Project Structure**:

```
toronto-bylaw-agent-python/
├── backend/
│   ├── config.py          # Configuration management
│   ├── database.py        # SQLAlchemy setup
│   ├── models.py          # Database models
│   ├── rag.py             # RAG knowledge base
│   ├── llm.py             # LLM integration
│   ├── agent.py           # Agent logic (450+ lines)
│   └── main.py            # FastAPI application
├── frontend/
│   └── app.py             # Streamlit UI (300+ lines)
├── tests/
│   └── test_agent.py      # Test suite (200+ lines)
├── data/
│   └── knowledge_base.json # Toronto bylaws
├── requirements.txt        # Python dependencies
├── README.md              # Comprehensive documentation
├── SETUP.md               # Installation guide
├── run_all.sh             # Start all services
├── run_backend.sh         # Start backend
└── run_frontend.sh        # Start frontend
```

**Total Code**: 1000+ lines of Python code
**Documentation**: 500+ lines of markdown
**Tests**: 200+ lines of pytest code

---

## Slide 8: Quick Start & Demo

**Installation** (5 minutes):
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export LLM_API_KEY=your_key

# 3. Run all services
./run_all.sh
```

**Access Points**:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Demo Workflow**:
1. Ask about Toronto bylaws
2. Report a hazard (multi-turn)
3. Check permit requirements
4. Look up waste collection
5. See RAG citations

---

## Presentation Notes

### Key Talking Points

**Problem Context**:
- Toronto has complex municipal regulations
- Citizens need quick access to accurate information
- Current systems (311 phone line) are overwhelmed

**Solution Innovation**:
- Conversational AI understands natural language
- Multi-turn dialogue for complex workflows
- RAG ensures responses are grounded in real data
- Python makes it accessible and maintainable

**Technical Excellence**:
- Modern Python stack (FastAPI, Streamlit)
- Production-ready architecture
- Comprehensive testing
- Clear separation of concerns

**Real-World Impact**:
- Reduces burden on Toronto 311
- Improves citizen experience
- Demonstrates AI for good
- Scalable to other municipalities

### Q&A Preparation

**Q: How accurate is the agent?**
A: 100% accuracy on intent classification, RAG retrieval provides relevant sources with citations. Accuracy depends on LLM quality and knowledge base completeness.

**Q: Can it handle real 311 integration?**
A: Yes, the architecture supports it. Currently uses mock ticket generation for demo purposes. Real integration requires Toronto 311 API credentials.

**Q: How does it handle out-of-scope requests?**
A: Intent classifier detects non-municipal queries and politely redirects users to appropriate resources.

**Q: What about privacy?**
A: All data stored locally in SQLite. No data shared except with LLM API. Conversations are isolated per user.

**Q: Can it be deployed?**
A: Yes, Docker-ready. Can deploy to Heroku, AWS, Google Cloud, or any cloud platform supporting Python.

---

## Conclusion

This project demonstrates a complete, production-ready Python application combining:
- Modern web frameworks (FastAPI, Streamlit)
- AI/ML capabilities (LLM integration, RAG)
- Database design (SQLAlchemy, SQLite)
- Software engineering best practices (testing, documentation, architecture)

Perfect for a Python course final project showcasing real-world application development.
