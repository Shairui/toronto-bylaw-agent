"""
config.py — Central configuration for the Toronto Bylaw Agent.

All settings are read from environment variables (.env locally, Streamlit Secrets in deployment).
Set GROQ_API_KEY to enable the LLM. USE_LLM is True when the key is present.

Embedding falls back to ChromaDB's built-in local model if the remote endpoint is unreachable.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
# PROJECT_ROOT is the repo root (one level above this file's directory).
PROJECT_ROOT = Path(__file__).parent.parent

# LLM
# Set GROQ_API_KEY in .env (local) or Streamlit Secrets (deployment).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

USE_LLM = bool(GROQ_API_KEY)

# RAG / ChromaDB
# CHROMA_DB_PATH: where the vector store is persisted on disk.
# KNOWLEDGE_BASE_PATH: optional external JSON knowledge base
#   (if the file doesn't exist, rag.py builds one from its hard-coded documents).
CHROMA_DB_PATH        = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "data" / "chroma_db"))
KNOWLEDGE_BASE_PATH   = os.getenv("KNOWLEDGE_BASE_PATH", str(PROJECT_ROOT / "data" / "knowledge_base.json"))

# Embedding
# Falls back to ChromaDB's built-in local model when the remote endpoint is unreachable.
EMBEDDING_API_KEY  = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://rsm-8430-a2.bjlkeng.io/v1")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Directory bootstrap
# Create data directories on first import so downstream code never has to.
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PROJECT_ROOT / "data", exist_ok=True)
