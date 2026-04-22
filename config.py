"""
config.py
─────────
Central configuration for the NotebookLM Replica.
All tuneable parameters live here so nothing is hard-coded elsewhere.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL          = "llama3.2:3b"        # Local Ollama model
EMBEDDING_MODEL    = "nomic-embed-text"   # Local Ollama embedding model

# ── Tavily Web Search ─────────────────────────────────────────────────────────
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")

# ── Document Processing ───────────────────────────────────────────────────────
CHUNK_SIZE         = 1000   # characters per chunk
CHUNK_OVERLAP      = 200    # overlap between chunks
TOP_K_RESULTS      = 4      # number of chunks to retrieve

# ── Storage Paths ─────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR         = os.path.join(BASE_DIR, "storage", "uploads")
CHROMA_DB_DIR      = os.path.join(BASE_DIR, "storage", "chroma_db")
NOTES_DIR          = os.path.join(BASE_DIR, "storage", "notes")

# Ensure all storage directories exist on startup
for directory in [UPLOAD_DIR, CHROMA_DB_DIR, NOTES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION  = "notebook_lm_docs"   # Collection name in ChromaDB

# ── UI Settings ───────────────────────────────────────────────────────────────
APP_TITLE          = "📓 NotebookLM Replica"
APP_ICON           = "📓"
MAX_NOTES_DISPLAY  = 20     # Max notes shown in the Notes Panel
