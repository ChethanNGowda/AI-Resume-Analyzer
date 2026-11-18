"""
Singleton Embedding Service for ResumeNova RAG pipeline.
Uses sentence-transformers ('all-MiniLM-L6-v2') on CPU to avoid GPU VRAM exhaustion on RTX 3050.
"""

import logging
import threading
from typing import List, Optional, Union
import numpy as np

from services.rag.config import RAGConfig

logger = logging.getLogger(__name__)

# Fallback deterministic pseudo-embedding dimension for test environments without model download
FALLBACK_DIMENSION = 384


class EmbeddingService:
    """
    Thread-safe Singleton wrapper for SentenceTransformer embedding generation.
    """
    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = RAGConfig.RAG_EMBEDDING_MODEL
        self.device = RAGConfig.EMBEDDING_DEVICE
        self._model = None
        self._init_failed = False
        self._init_error = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is not None or self._init_failed:
            return

        with self._lock:
            if self._model is not None or self._init_failed:
                return

            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
            except Exception as e:
                self._init_failed = True
                self._init_error = str(e)
                logger.warning(
                    f"Could not load SentenceTransformer ('{self.model_name}'): {e}. "
                    f"Falling back to deterministic pseudo-embeddings for testing/offline."
                )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of strings in batches.
        Returns float lists matching ChromaDB expectations.
        """
        if not texts:
            return []

        self._load_model()

        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error during model encoding: {e}. Using deterministic fallback.")

        # Fallback generator for tests or offline situations
        return [self._generate_fallback_embedding(t) for t in texts]

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic normalized 384-dimensional vector based on text content.
        Used as a robust fallback during offline testing or when HuggingFace is unreachable.
        """
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed pseudo-random generator with hash bytes
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.RandomState(seed)
        vec = rng.randn(FALLBACK_DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        if self._model is not None:
            if hasattr(self._model, "get_embedding_dimension"):
                return self._model.get_embedding_dimension()
            return self._model.get_sentence_embedding_dimension()
        return FALLBACK_DIMENSION
