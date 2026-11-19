"""
Configuration settings for the ResumeNova RAG pipeline.
Supports local inference via Ollama and sentence-transformers with configurable fallbacks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class RAGConfig:
    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    RAG_LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "qwen3.5:4b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Embeddings Configuration
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # RTX 3050 laptop GPU has 4GB VRAM. Use CPU for embeddings to preserve GPU memory for LLM
    EMBEDDING_DEVICE: str = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")

    # Vector Database Configuration
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() in ("true", "1", "yes")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.15"))

    # Vector DB storage directory
    _db_path = os.getenv("RAG_VECTOR_DB_PATH", "backend/data/chroma")
    if os.path.isabs(_db_path):
        VECTOR_DB_PATH: str = _db_path
    else:
        VECTOR_DB_PATH: str = str((BASE_DIR / _db_path.replace("backend/", "")).resolve())

    # Career knowledge documents path
    CAREER_KNOWLEDGE_DIR: str = str((BASE_DIR / "data" / "career_knowledge").resolve())

    # Maximum file upload size in bytes (10MB)
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

    # Supported file extensions
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".md"}

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "llm_model": cls.RAG_LLM_MODEL,
            "embedding_model": cls.RAG_EMBEDDING_MODEL,
            "embedding_device": cls.EMBEDDING_DEVICE,
            "rag_enabled": cls.RAG_ENABLED,
            "top_k": cls.RAG_TOP_K,
            "chunk_size": cls.RAG_CHUNK_SIZE,
            "chunk_overlap": cls.RAG_CHUNK_OVERLAP,
            "similarity_threshold": cls.RAG_SIMILARITY_THRESHOLD,
            "vector_db_path": cls.VECTOR_DB_PATH,
            "career_knowledge_dir": cls.CAREER_KNOWLEDGE_DIR,
        }
