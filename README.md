# Toronto 311 City Services Agent

A conversational AI assistant for Toronto municipal services, built for RSM8430 (University of Toronto, 2026). Handles hazard reporting, building permit screening, and waste disposal guidance — with multi-turn dialogue, RAG-powered knowledge retrieval, session conversation history, and guardrails against out-of-scope queries.

---

## File Structure

```
streamlit_app.py          ← Streamlit UI: fixed nav, CSS, session state, message
                            rendering, conversation history sidebar, quick-action cards
backend/
  config.py               ← All env vars loaded from .env / Streamlit Secrets
  llm.py                  ← Async LLM client using Groq (llama-3.3-70b-versatile)
  rag.py                  ← ChromaDB vector store. Builds 182-document knowledge base
                            from real data in excel/. Returns cosine distances for
                            relevance filtering. Falls back to keyword search when empty.
  agent.py                ← Intent classifier, guardrails, multi-turn handlers,
                            RAG relevance threshold, hazard ticket extraction
evaluation.py             ← 15-case evaluation script with optional LLM-as-judge
                            scoring via Claude (set ANTHROPIC_API_KEY to enable).
                            Writes results to data/
tests/
  test_agent.py           ← pytest suite: intent classification, guardrails,
                            multi-turn hazard flow, integration tests
excel/
  Waste Wizard Lookup Table.json          ← 2,206 official City of Toronto waste items
                                            (source for RAG waste documents)
  Cleared Building Permits since 2017.json ← 392,000 permit records
                                            (top 150 approved permits indexed in RAG)
  311_service_requests.csv                ← 500,000 311 service request records
data/
  chroma_db/              ← ChromaDB vector store (auto-created and wiped at startup)
  knowledge_base.json     ← Optional: place here to override the built-in documents
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**.

---

## Streamlit Cloud Deployment

1. Push the repo to GitHub (`.env` is gitignored — key is never committed)
2. In Streamlit Cloud → **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"
```

3. Set the main file to `streamlit_app.py`, Python 3.11

---

## Features

### Intent routing (5 categories)

| Intent | Example query |
|--------|--------------|
| `hazard_report` | "There's a pothole on King Street" |
| `permit_screener` | "Do I need a permit to add a deck?" |
| `collection_lookup` | "Which bin does a pizza box go in?" |
| `general_inquiry` | "What are the noise bylaws in Toronto?" |
| `out_of_scope` | "What's the weather?" / "Report in Hamilton" |

### Guardrails (run before LLM classification)

- **Prompt injection**: catches "ignore instructions", jailbreak attempts, DAN mode
- **Geographic**: refuses non-Toronto cities (Hamilton, Vancouver, Mississauga, etc.)
- **Topic**: refuses weather, politics, recipes, stock prices, etc.

### RAG with relevance filtering

- ChromaDB cosine similarity search over 182 documents built from real data sources
- Documents with distance ≥ 0.80 are dropped before being sent to the LLM
- Falls back to keyword search when ChromaDB is empty

### Multi-turn conversation

| Flow | Behaviour |
|------|-----------|
| **Hazard report** | LLM asks for missing location or hazard type; issues a mock `SR-2026-XXXXX` ticket when both are confirmed; ticket/location/hazard extracted and displayed in the action card |
| **Permit check** | Quick-action button opens a form (postal code + project description) |
| **General waste / bylaw** | Single-turn LLM answer grounded in RAG context |

### LLM backend: Groq

Default model: `llama-3.3-70b-versatile`. Override via `GROQ_MODEL` env var (e.g. `llama-3.1-8b-instant`).

---

## Evaluation

```bash
python evaluation.py
```

Runs 15 labelled test cases and scores:

| Metric | Description |
|--------|-------------|
| Intent accuracy | % where classified intent matches expected |
| Guardrail precision | % of out-of-scope queries correctly refused |
| Keyword hit rate | % of responses containing expected keywords |
| Citation rate | % of responses citing a source or toronto.ca |
| LLM quality score | Claude-as-judge score 1–5 (requires `ANTHROPIC_API_KEY`) |

Results saved to `data/evaluation_results.csv` and `data/evaluation_summary.csv`.

---

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | _(required)_ | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `ANTHROPIC_API_KEY` | _(optional)_ | Enables Claude-as-judge scoring in evaluation.py |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | Model used for LLM-as-judge evaluation |
| `CHROMA_DB_PATH` | `data/chroma_db` | ChromaDB persistence directory |
| `KNOWLEDGE_BASE_PATH` | `data/knowledge_base.json` | External KB (overrides built-in docs) |

---

## Course

RSM8430 · University of Toronto · 2026
