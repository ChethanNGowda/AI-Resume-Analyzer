"""
Personal Resume RAG Service for ResumeNova.
Provides user-isolated document ingestion, targeted retrieval, grounded question answering,
and recruiter introduction generation strictly from resume facts.
"""

import logging
import uuid
from typing import Dict, List, Optional

from services.rag.chunker import DocumentChunker
from services.rag.config import RAGConfig
from services.rag.document_parser import DocumentParser
from services.rag.llm_service import LLMService
from services.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class PersonalResumeRAG:
    """
    User-isolated Resume Retrieval-Augmented Generation service.
    """

    def __init__(self):
        self.vector_store = VectorStoreService.get_instance()
        self.llm_service = LLMService()
        self.chunker = DocumentChunker()

    def ingest_resume(
        self,
        user_id: str,
        file_input,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Parses, chunks, embeds, and stores user resume in an isolated Chroma collection.
        Returns ingestion metadata and extracted resume preview.
        """
        if not user_id:
            raise ValueError("user_id is required for personal resume ingestion.")

        # 1. Parse document
        parsed_doc = DocumentParser.parse_file(file_input, filename=filename)

        # 2. Generate unique source_id based on content hash
        source_id = f"resume_{parsed_doc['content_hash'][:16]}"

        # 3. Check for existing source for this user (prevent duplicate duplicate chunks)
        existing = self.vector_store.list_sources(user_id=user_id, collection_type="resume")
        for src in existing:
            if src.get("source_id") == source_id:
                return {
                    "success": True,
                    "source_id": source_id,
                    "document_name": parsed_doc["filename"],
                    "status": "already_indexed",
                    "chunk_count": src.get("chunk_count", 1),
                    "word_count": parsed_doc["word_count"],
                    "char_count": parsed_doc["char_count"],
                    "text_preview": parsed_doc["text"][:500],
                }

        # 4. Chunk document
        chunks = self.chunker.chunk_document(
            parsed_doc=parsed_doc,
            source_id=source_id,
            user_id=user_id,
            document_type="resume"
        )

        # 5. Ingest into user's isolated collection
        ingested_count = self.vector_store.ingest_chunks(
            user_id=user_id,
            chunks=chunks,
            collection_type="resume"
        )

        return {
            "success": True,
            "source_id": source_id,
            "document_name": parsed_doc["filename"],
            "status": "indexed",
            "chunk_count": ingested_count,
            "word_count": parsed_doc["word_count"],
            "char_count": parsed_doc["char_count"],
            "text_preview": parsed_doc["text"][:500],
            "raw_text": parsed_doc["text"],
        }

    def answer_resume_question(
        self,
        user_id: str,
        question: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Retrieves relevant resume chunks and generates a grounded response.
        If facts are not found in the resume, explicitly states they are absent.
        """
        if not user_id:
            return {
                "response": "Authentication required. Please sign in to query your private resume.",
                "citations": [],
                "grounded": False,
            }

        # Retrieve relevant chunks from user's isolated resume store
        chunks = self.vector_store.query(
            user_id=user_id,
            query_text=question,
            collection_type="resume",
            top_k=top_k or RAGConfig.RAG_TOP_K,
            similarity_threshold=0.25  # relaxed for conversational queries
        )

        if not chunks:
            # Check if user has uploaded any resume at all
            sources = self.vector_store.list_sources(user_id=user_id, collection_type="resume")
            if not sources:
                return {
                    "response": (
                        "No uploaded resume was found in your workspace. "
                        "Please upload your PDF or DOCX resume first so I can retrieve your specific details."
                    ),
                    "citations": [],
                    "grounded": False,
                }
            return {
                "response": (
                    "This information is not present in your uploaded resume. "
                    "I searched your resume documents, but found no matching qualifications or mentions."
                ),
                "citations": [],
                "grounded": True,
            }

        system_role = (
            "You are ResumeNova's Personal Resume Assistant.\n"
            "You are answering questions about the user's uploaded resume.\n"
            "STRICT FACTUAL GROUNDING RULES:\n"
            "1. Answer ONLY using facts directly stated in the retrieved resume context.\n"
            "2. If a specific skill, qualification, project, or date is not in the context, say explicitly: "
            "'This is not mentioned in your uploaded resume.'\n"
            "3. DO NOT hallucinate, infer, or fabricate any jobs, colleges, GPAs, or technical skills.\n"
            "4. Provide specific section or page citations with your answer.\n"
            "5. Keep the response organized, accurate, and professional."
        )

        return self.llm_service.generate_grounded_response(
            prompt=question,
            context_chunks=chunks,
            system_role=system_role
        )

    def generate_recruiter_introduction(
        self,
        user_id: str,
        target_role: Optional[str] = None
    ) -> Dict:
        """
        Generates a recruiter pitch / introduction letter strictly grounded in user resume facts.
        """
        # Query overview chunks
        chunks = self.vector_store.query(
            user_id=user_id,
            query_text="skills experience projects qualifications summary education",
            collection_type="resume",
            top_k=4
        )

        prompt = (
            f"Write a concise, professional recruiter pitch (120-160 words) introducing this candidate "
            f"for a {target_role or 'software engineering'} position.\n"
            f"Use ONLY their real skills, projects, and educational qualifications found in the retrieved resume.\n"
            f"Do not invent any achievements or companies."
        )

        system_role = (
            "You are an expert technical recruiter drafting a candidate profile pitch.\n"
            "Use only the provided resume facts. Do not embellish or add unverified claims."
        )

        return self.llm_service.generate_grounded_response(
            prompt=prompt,
            context_chunks=chunks,
            system_role=system_role
        )

    def get_resume_context_summary(self, user_id: str) -> Dict:
        """
        Retrieves top overview chunks to provide fast context indicator for the UI.
        """
        sources = self.vector_store.list_sources(user_id=user_id, collection_type="resume")
        return {
            "has_resume": len(sources) > 0,
            "document_count": len(sources),
            "sources": sources,
        }
