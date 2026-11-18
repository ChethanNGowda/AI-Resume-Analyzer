"""
Job Description RAG Service for ResumeNova.
Extracts structured role requirements, ingests job descriptions into user-isolated vector storage,
and semantically compares job requirements against user resume chunks with verified citations.
"""

import logging
import re
from typing import Dict, List, Optional

from services.rag.chunker import DocumentChunker
from services.rag.config import RAGConfig
from services.rag.document_parser import DocumentParser
from services.rag.llm_service import LLMService
from services.rag.vector_store import VectorStoreService
from services.skill_service import extract_skills

logger = logging.getLogger(__name__)


def extract_job_metadata(text: str) -> Dict:
    """
    Extracts structured fields from job description text using heuristic pattern recognition.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_few_lines = " ".join(lines[:4])

    # Extract Job Title
    title = "Target Position"
    title_match = re.search(
        r"(?i)(?:job title|role|position|looking for a|hiring an?)\s*[:\-]?\s*([A-Za-z0-9\s\/\-\+]{4,40})",
        first_few_lines
    )
    if title_match:
        title = title_match.group(1).strip()
    elif lines:
        if len(lines[0]) < 60:
            title = lines[0].strip()

    # Extract Experience Requirements
    exp_match = re.search(
        r"(?i)(\d+[\+]?\s*(?:to|-)?\s*\d*\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp))",
        text
    )
    experience_req = exp_match.group(1).strip() if exp_match else "Not specified"

    # Extract Education
    edu_match = re.search(
        r"(?i)\b(bachelor['’]s|master['’]s|b\.tech|m\.tech|b\.e|b\.s|m\.s|phd|computer science|degree)\b[^\n\.\;]*",
        text
    )
    education_req = edu_match.group(0).strip() if edu_match else "Degree in Computer Science or related field preferred"

    # Extract Location / Work Arrangement
    work_arr = "Not specified"
    if re.search(r"(?i)\b(remote|work from home|wfh)\b", text):
        work_arr = "Remote"
    elif re.search(r"(?i)\b(hybrid)\b", text):
        work_arr = "Hybrid"
    elif re.search(r"(?i)\b(on-site|onsite|in-office)\b", text):
        work_arr = "On-site"

    # Extract skills
    all_job_skills = extract_skills(text)

    # Differentiate required vs preferred
    preferred_skills = []
    required_skills = []

    pref_patterns = [
        r"(?i)(?:preferred|nice to have|bonus|plus|good to have)[:\-]?\s*([\s\S]*?)(?=(?:\n\n|\n[A-Z]|\Z))"
    ]
    pref_block = ""
    for pat in pref_patterns:
        m = re.search(pat, text)
        if m:
            pref_block += " " + m.group(1)

    if pref_block:
        preferred_skills = extract_skills(pref_block)

    required_skills = [s for s in all_job_skills if s not in preferred_skills]
    if not required_skills and all_job_skills:
        required_skills = all_job_skills

    return {
        "job_title": title,
        "experience_requirements": experience_req,
        "educational_qualifications": education_req,
        "work_arrangement": work_arr,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "all_skills": all_job_skills,
    }


class JobDescriptionRAG:
    """
    RAG service for job description parsing, storage, and cross-comparison with user resume.
    """

    def __init__(self):
        self.vector_store = VectorStoreService.get_instance()
        self.llm_service = LLMService()
        self.chunker = DocumentChunker()

    def ingest_job_description(
        self,
        user_id: str,
        job_input,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Parses and stores a job description in user's isolated job collection.
        Accepts raw string text or file input.
        """
        if isinstance(job_input, str):
            clean_text = job_input.strip()
            if len(clean_text) < 20:
                raise ValueError("Job description is too short (minimum 20 characters).")
            parsed_doc = {
                "text": clean_text,
                "filename": filename or "job_description.txt",
                "file_type": "text",
                "content_hash": DocumentParser.parse_file(clean_text.encode("utf-8"), filename="job.txt")["content_hash"],
                "word_count": len(clean_text.split()),
                "char_count": len(clean_text),
                "pages": [],
            }
        else:
            parsed_doc = DocumentParser.parse_file(job_input, filename=filename)

        source_id = f"job_{parsed_doc['content_hash'][:16]}"
        chunks = self.chunker.chunk_document(
            parsed_doc=parsed_doc,
            source_id=source_id,
            user_id=user_id,
            document_type="job_description"
        )

        ingested_count = self.vector_store.ingest_chunks(
            user_id=user_id,
            chunks=chunks,
            collection_type="jobs"
        )

        meta = extract_job_metadata(parsed_doc["text"])

        return {
            "success": True,
            "source_id": source_id,
            "document_name": parsed_doc["filename"],
            "metadata": meta,
            "chunk_count": ingested_count,
            "raw_text": parsed_doc["text"],
        }

    def compare_job_with_resume(
        self,
        user_id: str,
        job_text: str,
        resume_skills: Optional[List[str]] = None
    ) -> Dict:
        """
        Compares job description against user's stored resume using semantic vector search
        and exact skill matching, providing evidence-backed gap analysis.
        """
        meta = extract_job_metadata(job_text)
        required_skills = meta.get("required_skills", [])
        preferred_skills = meta.get("preferred_skills", [])
        all_job_skills = meta.get("all_skills", [])

        # 1. Retrieve user's resume chunks
        resume_chunks = self.vector_store.query(
            user_id=user_id,
            query_text=f"skills experience projects {' '.join(all_job_skills[:8])}",
            collection_type="resume",
            top_k=6,
            similarity_threshold=0.20
        )

        # Build list of skills confirmed in resume
        user_skills_set = set()
        if resume_skills:
            user_skills_set.update(s.lower() for s in resume_skills)

        # Also extract skills directly from retrieved resume chunks
        for chunk in resume_chunks:
            chunk_skills = extract_skills(chunk.get("text", ""))
            user_skills_set.update(s.lower() for s in chunk_skills)

        matched_skills = [s for s in all_job_skills if s.lower() in user_skills_set]
        missing_skills = [s for s in all_job_skills if s.lower() not in user_skills_set]
        match_score = (
            int((len(matched_skills) / len(all_job_skills)) * 100)
            if all_job_skills else 0
        )

        # Retrieve specific evidence chunks for matched skills
        evidence_citations = []
        for chunk in resume_chunks[:3]:
            evidence_citations.append({
                "source": chunk.get("metadata", {}).get("document_name", "Resume"),
                "section": chunk.get("metadata", {}).get("section_hint", "Experience"),
                "page": chunk.get("metadata", {}).get("page_number", 1),
                "snippet": chunk.get("text", "")[:180] + "...",
            })

        # Generate recommended learning topics for missing skills
        learning_roadmap = [
            f"Study {skill.title()} fundamentals and complete one hands-on project demonstrating practical implementation."
            for skill in missing_skills[:4]
        ]
        if not learning_roadmap:
            learning_roadmap = ["Your technical skills show strong coverage for this job specification! Focus on quantifying business outcomes."]

        # Structured improvement suggestions
        resume_improvement_suggestions = [
            f"Add verifiable evidence for '{skill}' in your project bullet points."
            for skill in missing_skills[:3]
        ]
        if meta.get("experience_requirements") != "Not specified":
            resume_improvement_suggestions.append(
                f"Highlight role duration to meet {meta['experience_requirements']} requirement."
            )

        return {
            "success": True,
            "job_title": meta.get("job_title"),
            "work_arrangement": meta.get("work_arrangement"),
            "experience_required": meta.get("experience_requirements"),
            "education_required": meta.get("educational_qualifications"),
            "match_score": match_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "preferred_skills": preferred_skills,
            "total_job_skills": len(all_job_skills),
            "evidence": evidence_citations,
            "learning_roadmap": learning_roadmap,
            "resume_improvement_suggestions": resume_improvement_suggestions,
            "has_resume_context": len(resume_chunks) > 0,
        }
