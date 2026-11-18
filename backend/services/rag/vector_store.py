"""
ChromaDB Vector Store Service for ResumeNova.
Enforces strict user isolation, metadata preservation, deduplication, and graceful error handling.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from services.rag.config import RAGConfig
from services.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def sanitize_collection_name(name: str) -> str:
    """
    ChromaDB collection name rules: 3-63 chars, alphanumeric, dashes, underscores.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
    if len(clean) < 3:
        clean = f"col_{clean}"
    return clean[:63]


class VectorStoreService:
    """
    Singleton ChromaDB client manager providing user-isolated document storage and retrieval.
    """
    _instance: Optional["VectorStoreService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.db_path = RAGConfig.VECTOR_DB_PATH
        self.embedding_service = EmbeddingService.get_instance()
        self._client = None
        self._is_available = False
        self._init_client()

    @classmethod
    def get_instance(cls) -> "VectorStoreService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_client(self):
        """Initialize ChromaDB PersistentClient."""
        if chromadb is None:
            logger.warning("chromadb package not installed. Vector store disabled.")
            self._is_available = False
            return

        try:
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            self._is_available = True
            logger.info(f"ChromaDB persistent client initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at {self.db_path}: {e}")
            self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available and self._client is not None

    def _get_user_collection(self, user_id: str, collection_type: str = "resume"):
        """
        Retrieves or creates an isolated collection for a specific user.
        Format: usr_{type}_{sanitized_user_id}
        """
        if not self.is_available:
            raise RuntimeError("Vector database is not available.")

        safe_user = sanitize_collection_name(user_id or "guest")
        col_name = sanitize_collection_name(f"usr_{collection_type}_{safe_user}")
        return self._client.get_or_create_collection(
            name=col_name,
            metadata={"user_id": user_id, "type": collection_type, "hnsw:space": "cosine"}
        )

    def _get_career_collection(self):
        """Shared vetted career knowledge collection (no private user data)."""
        if not self.is_available:
            raise RuntimeError("Vector database is not available.")
        return self._client.get_or_create_collection(
            name="resumenova_career_knowledge",
            metadata={"type": "career_knowledge", "hnsw:space": "cosine"}
        )

    def ingest_chunks(
        self,
        user_id: str,
        chunks: List[Dict],
        collection_type: str = "resume"
    ) -> int:
        """
        Embed and store document chunks in user-isolated collection.
        Skips duplicate chunks if already present.
        Returns count of ingested chunks.
        """
        if not chunks:
            return 0
        if not self.is_available:
            logger.warning("Vector database unavailable. Skipping chunk ingestion.")
            return 0

        # Validate user isolation
        for chunk in chunks:
            if chunk.get("user_id") != user_id and collection_type != "career_knowledge":
                raise ValueError("Unauthorized: Chunk user_id does not match target collection owner.")

        collection = (
            self._get_career_collection()
            if collection_type == "career_knowledge"
            else self._get_user_collection(user_id, collection_type)
        )

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        metadatas = []

        for c in chunks:
            # Metadata must be primitive types for ChromaDB
            meta = {
                "source_id": str(c.get("source_id", "")),
                "document_name": str(c.get("document_name", "")),
                "document_type": str(c.get("document_type", collection_type)),
                "user_id": str(user_id),
                "chunk_index": int(c.get("chunk_index", 0)),
                "total_chunks": int(c.get("total_chunks", len(chunks))),
                "page_number": int(c.get("page_number", 1)),
                "section_hint": str(c.get("section_hint", "General")),
                "char_count": int(c.get("char_count", len(c.get("text", "")))),
            }
            metadatas.append(meta)

        # Generate embeddings
        embeddings = self.embedding_service.embed_batch(texts)

        # Upsert into ChromaDB
        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Ingested {len(chunks)} chunks for user '{user_id}' in {collection_type} collection.")
        return len(chunks)

    def query(
        self,
        user_id: str,
        query_text: str,
        collection_type: str = "resume",
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Perform semantic search against user's collection or shared career knowledge.
        Strictly filters by user_id to enforce data isolation.
        """
        if not self.is_available or not query_text.strip():
            return []

        k = top_k or RAGConfig.RAG_TOP_K
        threshold = similarity_threshold or RAGConfig.RAG_SIMILARITY_THRESHOLD

        try:
            collection = (
                self._get_career_collection()
                if collection_type == "career_knowledge"
                else self._get_user_collection(user_id, collection_type)
            )

            if collection.count() == 0:
                return []

            query_embedding = self.embedding_service.embed_text(query_text)

            where_filter = None
            if collection_type != "career_knowledge" and filter_metadata:
                and_conds = [{"user_id": {"$eq": str(user_id)}}]
                for k_f, v_f in filter_metadata.items():
                    and_conds.append({k_f: {"$eq": v_f}})
                where_filter = {"$and": and_conds}
            elif collection_type != "career_knowledge":
                where_filter = {"user_id": str(user_id)}
            elif filter_metadata:
                where_filter = filter_metadata

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            hits = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

                for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                    # Chroma cosine distance: cosine_sim = 1 - distance
                    # For normalized embeddings with cosine space, distance in [0, 2]
                    similarity = round(max(0.0, 1.0 - float(dist)), 4)
                    if similarity >= threshold:
                        hits.append({
                            "chunk_id": cid,
                            "text": doc,
                            "metadata": meta,
                            "similarity": similarity,
                            "distance": round(float(dist), 4),
                        })

            # Sort by similarity descending
            hits.sort(key=lambda x: x["similarity"], reverse=True)
            return hits
        except Exception as e:
            logger.error(f"Error querying vector store for user '{user_id}': {e}")
            return []

    def delete_source(self, user_id: str, source_id: str, collection_type: str = "resume") -> bool:
        """
        Delete all chunks belonging to a specific document for a user.
        Ensures user ownership before deletion.
        """
        if not self.is_available:
            return False

        try:
            collection = (
                self._get_career_collection()
                if collection_type == "career_knowledge"
                else self._get_user_collection(user_id, collection_type)
            )

            # Query items with source_id and verify user isolation
            if collection_type != "career_knowledge":
                where_condition = {
                    "$and": [
                        {"source_id": {"$eq": str(source_id)}},
                        {"user_id": {"$eq": str(user_id)}}
                    ]
                }
            else:
                where_condition = {"source_id": str(source_id)}

            existing = collection.get(where=where_condition)
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
                logger.info(f"Deleted {len(existing['ids'])} chunks for source '{source_id}', user '{user_id}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting source '{source_id}' for user '{user_id}': {e}")
            return False

    def list_sources(self, user_id: str, collection_type: str = "resume") -> List[Dict]:
        """
        List all distinct documents ingested for a user.
        """
        if not self.is_available:
            return []

        try:
            collection = (
                self._get_career_collection()
                if collection_type == "career_knowledge"
                else self._get_user_collection(user_id, collection_type)
            )

            if collection.count() == 0:
                return []

            where_condition = None if collection_type == "career_knowledge" else {"user_id": str(user_id)}
            data = collection.get(where=where_condition, include=["metadatas"])

            sources_dict = {}
            if data and data.get("metadatas"):
                for m in data["metadatas"]:
                    sid = m.get("source_id")
                    if sid and sid not in sources_dict:
                        sources_dict[sid] = {
                            "source_id": sid,
                            "document_name": m.get("document_name", "Document"),
                            "document_type": m.get("document_type", collection_type),
                            "chunk_count": m.get("total_chunks", 1),
                            "user_id": m.get("user_id", user_id),
                        }

            return list(sources_dict.values())
        except Exception as e:
            logger.error(f"Error listing sources for user '{user_id}': {e}")
            return []

    def clear_user_collection(self, user_id: str, collection_type: str = "resume") -> bool:
        """Completely reset a user's isolated collection."""
        if not self.is_available:
            return False

        try:
            safe_user = sanitize_collection_name(user_id or "guest")
            col_name = sanitize_collection_name(f"usr_{collection_type}_{safe_user}")
            try:
                self._client.delete_collection(name=col_name)
                logger.info(f"Deleted collection '{col_name}' for user '{user_id}'")
                return True
            except Exception:
                return False
        except Exception as e:
            logger.error(f"Error clearing collection for user '{user_id}': {e}")
            return False

    def get_status(self) -> Dict:
        """Returns health and status of the vector database."""
        status = {
            "available": self.is_available,
            "engine": "ChromaDB",
            "storage_path": self.db_path,
            "embedding_model": RAGConfig.RAG_EMBEDDING_MODEL,
            "embedding_dimension": self.embedding_service.dimension,
            "collections_count": 0,
        }
        if self.is_available:
            try:
                cols = self._client.list_collections()
                status["collections_count"] = len(cols)
            except Exception as e:
                status["error"] = str(e)
        return status
