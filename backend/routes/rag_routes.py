"""
RAG API Endpoints for ResumeNova.
Provides secure, user-isolated chat, document ingestion, job gap analysis,
interview preparation, source management, and status monitoring.
"""

import logging
from flask import Blueprint, jsonify, request, session
from security import require_login

from services.rag.assistant_service import AssistantService
from services.rag.career_knowledge import CareerKnowledgeService
from services.rag.config import RAGConfig
from services.rag.document_parser import DocumentParseError
from services.rag.job_rag import JobDescriptionRAG
from services.rag.llm_service import LLMService
from services.rag.resume_rag import PersonalResumeRAG
from services.rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

rag_bp = Blueprint("rag", __name__)

assistant_service = AssistantService()
resume_rag = PersonalResumeRAG()
job_rag = JobDescriptionRAG()
career_service = CareerKnowledgeService()
try:
    career_service.initialize_knowledge_base()
except Exception as _e:
    logger.warning(f"Could not auto-initialize career knowledge base: {_e}")
vector_store = VectorStoreService.get_instance()
llm_service = LLMService()


def get_current_user_id() -> str:
    """Return the identity established by the signed server-side session."""
    user_id = session.get("user_id")
    if not user_id:
        raise RuntimeError("Authentication required.")
    return user_id


@rag_bp.route("/status", methods=["GET"])
def rag_status():
    """
    Health check and status of RAG subsystems (Ollama LLM, ChromaDB vector store, Embeddings).
    """
    ollama_health = llm_service.check_health()
    vdb_status = vector_store.get_status()

    return jsonify({
        "success": True,
        "rag_enabled": RAGConfig.RAG_ENABLED,
        "llm": ollama_health,
        "vector_database": vdb_status,
        "config": {
            "model": RAGConfig.RAG_LLM_MODEL,
            "embedding_model": RAGConfig.RAG_EMBEDDING_MODEL,
            "top_k": RAGConfig.RAG_TOP_K,
            "chunk_size": RAGConfig.RAG_CHUNK_SIZE,
            "chunk_overlap": RAGConfig.RAG_CHUNK_OVERLAP,
        }
    })


@rag_bp.route("/ingest", methods=["POST"])
@require_login
def ingest_document():
    """
    Ingest a resume or job description file (PDF, DOCX, TXT, MD) into user-isolated vector store.
    """
    user_id = get_current_user_id()
    doc_type = request.form.get("type", "resume").lower()

    if "file" not in request.files:
        # Check if raw text was passed in JSON
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            raw_text = payload.get("text", "").strip()
            filename = payload.get("filename", "document.txt")
            if not raw_text:
                return jsonify({"error": "No file or text payload provided."}), 400

            try:
                if doc_type == "job_description":
                    result = job_rag.ingest_job_description(user_id=user_id, job_input=raw_text, filename=filename)
                else:
                    result = resume_rag.ingest_resume(user_id=user_id, file_input=raw_text.encode("utf-8"), filename=filename)
                return jsonify({"success": True, "data": result}), 201
            except Exception as e:
                return jsonify({"error": str(e)}), 400

        return jsonify({"error": "No file uploaded. Please include 'file' in multipart form data."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Selected file has an empty filename."}), 400

    try:
        if doc_type in ("job_description", "job", "jobs"):
            result = job_rag.ingest_job_description(
                user_id=user_id,
                job_input=uploaded_file,
                filename=uploaded_file.filename
            )
        else:
            result = resume_rag.ingest_resume(
                user_id=user_id,
                file_input=uploaded_file,
                filename=uploaded_file.filename
            )

        return jsonify({"success": True, "data": result}), 201
    except DocumentParseError as de:
        return jsonify({"error": str(de)}), 400
    except Exception as e:
        logger.error(f"Ingestion error for user '{user_id}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to ingest document: {str(e)}"}), 500


@rag_bp.route("/chat", methods=["POST"])
@require_login
def rag_chat():
    """
    Conversational AI Assistant endpoint.
    Retrieves grounded evidence from isolated resume, job, or career knowledge collections.
    """
    user_id = get_current_user_id()
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()
    active_context = payload.get("context", "auto")

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        response = assistant_service.chat(
            user_id=user_id,
            message=message,
            active_context=active_context
        )
        return jsonify({
            "success": True,
            "data": response,
            "user_id": user_id,
        })
    except Exception as e:
        logger.error(f"Chat error for user '{user_id}': {e}", exc_info=True)
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


@rag_bp.route("/sources", methods=["GET"])
@require_login
def list_user_sources():
    """
    List all ingested documents for the authenticated user.
    """
    user_id = get_current_user_id()
    resume_sources = vector_store.list_sources(user_id=user_id, collection_type="resume")
    job_sources = vector_store.list_sources(user_id=user_id, collection_type="jobs")

    return jsonify({
        "success": True,
        "user_id": user_id,
        "resumes": resume_sources,
        "jobs": job_sources,
        "total": len(resume_sources) + len(job_sources),
    })


@rag_bp.route("/sources/<source_id>", methods=["DELETE"])
@require_login
def delete_user_source(source_id):
    """
    Delete a specific document and its vector embeddings for the current user.
    """
    user_id = get_current_user_id()
    col_type = request.args.get("type", "resume")

    deleted = vector_store.delete_source(user_id=user_id, source_id=source_id, collection_type=col_type)
    if not deleted and col_type == "resume":
        # Attempt deleting from jobs if not found in resume
        deleted = vector_store.delete_source(user_id=user_id, source_id=source_id, collection_type="jobs")

    if deleted:
        return jsonify({"success": True, "message": f"Source '{source_id}' removed successfully."})
    return jsonify({"error": "Source not found or unauthorized."}), 404


@rag_bp.route("/job-analysis", methods=["POST"])
@require_login
def job_analysis():
    """
    Compare job description against user resume with evidence-backed citations.
    """
    user_id = get_current_user_id()
    payload = request.get_json(silent=True) or {}
    job_description = payload.get("job_description", "").strip()
    resume_skills = payload.get("resume_skills") or []

    if not job_description:
        return jsonify({"error": "job_description is required."}), 400

    try:
        comparison = job_rag.compare_job_with_resume(
            user_id=user_id,
            job_text=job_description,
            resume_skills=resume_skills
        )
        return jsonify({"success": True, "data": comparison})
    except Exception as e:
        logger.error(f"Job analysis error: {e}", exc_info=True)
        return jsonify({"error": f"Job analysis failed: {str(e)}"}), 500


@rag_bp.route("/interview-preparation", methods=["POST"])
@require_login
def interview_preparation():
    """
    Generates interview questions grounded in the user's actual projects, skills, and target role.
    """
    user_id = get_current_user_id()
    payload = request.get_json(silent=True) or {}
    target_role = payload.get("target_role", "Software Engineer")

    # Retrieve real resume projects and skills
    resume_chunks = vector_store.query(
        user_id=user_id,
        query_text="projects technical skills achievements",
        collection_type="resume",
        top_k=4
    )

    career_chunks = career_service.search_career_knowledge(
        query=f"technical interview questions for {target_role}",
        top_k=3
    )

    all_chunks = resume_chunks + career_chunks

    prompt = (
        f"Generate 5 targeted interview questions for a {target_role} position based on this candidate's "
        f"actual projects, tools, and technical experience retrieved from their resume.\n"
        f"Include suggested answer talking points grounded in their real resume evidence."
    )

    response = llm_service.generate_grounded_response(
        prompt=prompt,
        context_chunks=all_chunks,
        system_role="You are a senior engineering hiring manager conducting a technical interview."
    )

    return jsonify({
        "success": True,
        "target_role": target_role,
        "data": response,
        "has_resume_evidence": len(resume_chunks) > 0,
    })


@rag_bp.route("/clear-conversation", methods=["POST"])
@require_login
def clear_conversation():
    """Clear chat history for user session."""
    user_id = get_current_user_id()
    assistant_service.clear_conversation(user_id=user_id)
    return jsonify({"success": True, "message": "Conversation cleared."})


@rag_bp.route("/career-topics", methods=["GET"])
@require_login
def list_career_topics():
    """List available curated career knowledge topics."""
    topics = career_service.list_topics()
    return jsonify({"success": True, "topics": topics})
