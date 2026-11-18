"""
Career Knowledge Base Manager for ResumeNova.
Ingests and indexes trusted career guidance, interview preparation, and technical reference documents into ChromaDB.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from services.rag.chunker import DocumentChunker
from services.rag.config import RAGConfig
from services.rag.document_parser import DocumentParser
from services.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class CareerKnowledgeService:
    """
    Manages curated career and interview preparation reference documents.
    """

    def __init__(self):
        self.knowledge_dir = Path(RAGConfig.CAREER_KNOWLEDGE_DIR)
        self.vector_store = VectorStoreService.get_instance()
        self.chunker = DocumentChunker(chunk_size=800, chunk_overlap=120)

    def initialize_knowledge_base(self, force: bool = False) -> Dict:
        """
        Scans knowledge directory and ingests all markdown and text guides.
        """
        if not self.knowledge_dir.exists():
            logger.warning(f"Career knowledge directory not found: {self.knowledge_dir}")
            return {"status": "directory_missing", "ingested": 0}

        if not self.vector_store.is_available:
            return {"status": "vector_store_unavailable", "ingested": 0}

        existing_sources = self.vector_store.list_sources("career_admin", collection_type="career_knowledge")
        existing_filenames = {s.get("document_name") for s in existing_sources}

        files_to_process = list(self.knowledge_dir.glob("*.md")) + list(self.knowledge_dir.glob("*.txt"))
        total_chunks_ingested = 0
        documents_processed = 0

        for file_path in files_to_process:
            if not force and file_path.name in existing_filenames:
                continue

            try:
                parsed = DocumentParser.parse_file(file_path)
                source_id = f"career_{file_path.stem}"
                chunks = self.chunker.chunk_document(
                    parsed_doc=parsed,
                    source_id=source_id,
                    user_id="career_admin",
                    document_type="career_knowledge"
                )

                if chunks:
                    self.vector_store.ingest_chunks(
                        user_id="career_admin",
                        chunks=chunks,
                        collection_type="career_knowledge"
                    )
                    total_chunks_ingested += len(chunks)
                    documents_processed += 1
            except Exception as e:
                logger.error(f"Failed to ingest career document {file_path.name}: {e}")

        logger.info(f"Career knowledge base initialized: {documents_processed} files, {total_chunks_ingested} chunks.")
        return {
            "status": "success",
            "documents_processed": documents_processed,
            "total_chunks": total_chunks_ingested,
            "total_files": len(files_to_process),
        }

    def search_career_knowledge(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Query the curated career knowledge base."""
        return self.vector_store.query(
            user_id="career_admin",
            query_text=query,
            collection_type="career_knowledge",
            top_k=top_k or RAGConfig.RAG_TOP_K,
            similarity_threshold=similarity_threshold or RAGConfig.RAG_SIMILARITY_THRESHOLD
        )

    def list_topics(self) -> List[Dict]:
        """List all available career knowledge guide topics."""
        if not self.knowledge_dir.exists():
            return []

        files = list(self.knowledge_dir.glob("*.md")) + list(self.knowledge_dir.glob("*.txt"))
        topics = []
        for f in sorted(files):
            title = f.stem.replace("_", " ").title()
            topics.append({
                "id": f.stem,
                "title": title,
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            })
        return topics
