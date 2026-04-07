"""Configuration management for Toronto Bylaw Agent."""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "https://forge.manus.im/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-30b-a3b-fp8")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./toronto_bylaw.db")

# Server
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

# RAG
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "data" / "chroma_db"))
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", str(PROJECT_ROOT / "data" / "knowledge_base.json"))

# Ensure directories exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PROJECT_ROOT / "data", exist_ok=True)
