"""
Conversational AI Career Assistant for ResumeNova.
Orchestrates multi-turn conversation, intelligent intent routing between personal resume,
job descriptions, and career knowledge bases, and formats evidence-grounded responses with citations.
"""

import logging
from typing import Dict, List, Optional

from services.rag.career_knowledge import CareerKnowledgeService
from services.rag.config import RAGConfig
from services.rag.job_rag import JobDescriptionRAG
from services.rag.llm_service import LLMService
from services.rag.resume_rag import PersonalResumeRAG
from services.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# In-memory conversation session store (user_id -> List[Dict])
CONVERSATION_SESSIONS: Dict[str, List[Dict]] = {}


class AssistantService:
    """
    Main conversational agent routing queries to appropriate RAG indices.
    """

    def __init__(self):
        self.resume_rag = PersonalResumeRAG()
        self.job_rag = JobDescriptionRAG()
        self.career_service = CareerKnowledgeService()
        self.vector_store = VectorStoreService.get_instance()
        self.llm_service = LLMService()

    def chat(
        self,
        user_id: str,
        message: str,
        active_context: Optional[str] = None  # 'auto', 'resume', 'job', 'career'
    ) -> Dict:
        """
        Processes a user message, retrieves evidence from appropriate collections,
        and generates an evidence-grounded response.
        """
        clean_msg = message.strip()
        if not clean_msg:
            return {
                "response": "Please enter a question or prompt.",
                "citations": [],
                "context_used": "none",
                "grounded": False,
            }

        # Initialize conversation session if needed
        if user_id not in CONVERSATION_SESSIONS:
            CONVERSATION_SESSIONS[user_id] = []

        # Detect intent and route to target knowledge collection
        intent = self._detect_intent(clean_msg, active_context)
        context_chunks: List[Dict] = []
        context_used = intent

        if intent == "resume":
            context_chunks = self.vector_store.query(
                user_id=user_id,
                query_text=clean_msg,
                collection_type="resume",
                top_k=RAGConfig.RAG_TOP_K,
                similarity_threshold=0.20
            )
            # If resume is empty, fall back to checking if user has any uploaded resume
            if not context_chunks:
                sources = self.vector_store.list_sources(user_id=user_id, collection_type="resume")
                if not sources:
                    context_used = "general_fallback"

        elif intent == "job":
            context_chunks = self.vector_store.query(
                user_id=user_id,
                query_text=clean_msg,
                collection_type="jobs",
                top_k=RAGConfig.RAG_TOP_K,
                similarity_threshold=0.20
            )

        elif intent == "career":
            context_chunks = self.career_service.search_career_knowledge(
                query=clean_msg,
                top_k=RAGConfig.RAG_TOP_K,
                similarity_threshold=0.20
            )

        else:  # hybrid / general
            # Query both resume and career knowledge
            resume_hits = self.vector_store.query(
                user_id=user_id,
                query_text=clean_msg,
                collection_type="resume",
                top_k=3,
                similarity_threshold=0.22
            )
            career_hits = self.career_service.search_career_knowledge(
                query=clean_msg,
                top_k=3,
                similarity_threshold=0.22
            )
            context_chunks = resume_hits + career_hits
            context_used = "hybrid"

        # Check if we have sufficient evidence
        if not context_chunks and context_used == "resume":
            sources = self.vector_store.list_sources(user_id=user_id, collection_type="resume")
            if not sources:
                answer = (
                    "No resume has been uploaded to your workspace yet. "
                    "Please upload your resume PDF in the Dashboard so I can answer questions about your experience, "
                    "skills, and projects."
                )
            else:
                answer = (
                    "I searched your uploaded resume, but could not find information addressing this question. "
                    "This skill or experience does not appear to be listed in your resume."
                )
            result = {
                "response": answer,
                "citations": [],
                "context_used": "resume",
                "grounded": True,
            }
        else:
            # Generate grounded response
            system_role = self._build_system_role(intent)
            result = self.llm_service.generate_grounded_response(
                prompt=clean_msg,
                context_chunks=context_chunks,
                system_role=system_role
            )
            result["context_used"] = context_used

        # Record conversation turn
        CONVERSATION_SESSIONS[user_id].append({"role": "user", "content": clean_msg})
        CONVERSATION_SESSIONS[user_id].append({"role": "assistant", "content": result["response"]})

        # Keep history bounded (last 16 messages)
        if len(CONVERSATION_SESSIONS[user_id]) > 16:
            CONVERSATION_SESSIONS[user_id] = CONVERSATION_SESSIONS[user_id][-16:]

        return result

    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Returns message history for user."""
        return CONVERSATION_SESSIONS.get(user_id, [])

    def clear_conversation(self, user_id: str) -> bool:
        """Clears user's current session history."""
        CONVERSATION_SESSIONS[user_id] = []
        return True

    def _detect_intent(self, message: str, explicit_context: Optional[str] = None) -> str:
        """
        Classifies user query into 'resume', 'job', 'career', or 'hybrid'.
        """
        if explicit_context in ("resume", "job", "career"):
            return explicit_context

        lower = message.lower()

        # Job description intent cues
        if any(w in lower for w in [
            "job description", "target role", "job requirements", "fit for this role",
            "am i qualified", "job match", "missing for this job"
        ]):
            return "job"

        # Personal resume intent cues
        if any(w in lower for w in [
            "my resume", "my skills", "my project", "my education", "my degree",
            "my gpa", "my experience", "what do i have", "recruiter pitch for me",
            "introduce me", "my strengths", "my weaknesses"
        ]):
            return "resume"

        # Career knowledge intent cues
        if any(w in lower for w in [
            "how to prepare", "system design", "star method", "solid principles",
            "interview tips", "ats optimization", "how does ats work",
            "placement prep", "dsa roadmap", "behavioral question", "behavioral",
            "interview questions", "interview question", "salary negotiation",
            "hr interview", "technical interview", "coding interview",
        ]):
            return "career"

        # Default to hybrid search across user resume and career knowledge
        return "hybrid"

    def _build_system_role(self, intent: str) -> str:
        """Constructs persona and strict grounding directives based on intent."""
        base = (
            "You are ResumeNova's AI Career Assistant and Resume Coach.\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. Ground every answer in the retrieved context sources provided below.\n"
            "2. Whenever mentioning a fact, cite the document name and section.\n"
            "3. If facts are missing from the retrieved context, state clearly: "
            "'This information is not present in the provided documents.'\n"
            "4. NEVER fabricate qualifications, experience, companies, marks, or dates.\n"
            "5. Keep responses structured, professional, and directly actionable."
        )

        if intent == "resume":
            return base + "\nFocus on extracting and summarizing real qualifications from the user's resume."
        elif intent == "job":
            return base + "\nFocus on comparing the job requirements against user resume evidence, highlighting matches and gaps honestly."
        elif intent == "career":
            return base + "\nFocus on sharing trusted software engineering, interview preparation, and placement strategies from the reference documents."
        return base
