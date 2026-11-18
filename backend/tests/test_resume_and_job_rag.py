"""
Tests for Personal Resume RAG and Job Description RAG.
Verifies grounded Q&A, evidence citations, missing skill identification, and missing context handling.
"""

import pytest
from services.rag.job_rag import JobDescriptionRAG, extract_job_metadata
from services.rag.resume_rag import PersonalResumeRAG
from services.rag.vector_store import VectorStoreService


@pytest.fixture
def rag_services():
    return PersonalResumeRAG(), JobDescriptionRAG()


def test_extract_job_metadata():
    jd_text = (
        "Role: Senior Backend Engineer\n"
        "Experience: 4+ years of experience\n"
        "Requirements: Bachelor's degree in Computer Science.\n"
        "Work arrangement: Remote\n"
        "Must have skills: Python, Flask, PostgreSQL, Docker, REST API\n"
        "Preferred skills: Kubernetes, AWS"
    )
    meta = extract_job_metadata(jd_text)
    assert "Senior Backend Engineer" in meta["job_title"]
    assert "4+ years" in meta["experience_requirements"]
    assert meta["work_arrangement"] == "Remote"
    assert "python" in [s.lower() for s in meta["required_skills"]]
    assert "flask" in [s.lower() for s in meta["required_skills"]]


def test_resume_grounded_qa_and_missing_facts(rag_services):
    resume_rag, _ = rag_services
    user_id = "test_user_grounding"

    resume_text = (
        "Jane Doe\n"
        "Education: Bachelor of Technology in Computer Engineering, graduated 2024.\n"
        "Technical Skills: Python, Flask, SQL, pandas, Git\n"
        "Projects: Built Resume Analyzer web application with Flask and React."
    )

    ingest_result = resume_rag.ingest_resume(
        user_id=user_id,
        file_input=resume_text.encode("utf-8"),
        filename="jane_resume.txt"
    )
    assert ingest_result["success"] is True

    # 1. Ask about technical skills present
    skills_answer = resume_rag.answer_resume_question(user_id, "What technical skills do I have?")
    assert skills_answer["grounded"] is True
    assert "jane_resume.txt" in str(skills_answer.get("citations", [])) or "Python" in skills_answer["response"]

    # 2. Ask about something NOT in the resume (e.g. Fortran, Doctorate, AWS)
    missing_answer = resume_rag.answer_resume_question(user_id, "What is my experience with Fortran or Cobol?")
    assert (
        "not present" in missing_answer["response"].lower()
        or "not mentioned" in missing_answer["response"].lower()
        or "could not find" in missing_answer["response"].lower()
    )

    # 3. Test recruiter pitch generation
    pitch = resume_rag.generate_recruiter_introduction(user_id, target_role="Backend Developer")
    assert pitch["grounded"] is True
    assert len(pitch["response"]) > 30

    # Cleanup
    VectorStoreService.get_instance().clear_user_collection(user_id, "resume")


def test_job_description_comparison_with_resume(rag_services):
    resume_rag, job_rag = rag_services
    user_id = "test_user_job_match"

    resume_text = (
        "Technical Skills: Python, Flask, SQL, Git, REST API\n"
        "Projects: Developed REST API using Python and Flask."
    )
    resume_rag.ingest_resume(
        user_id=user_id,
        file_input=resume_text.encode("utf-8"),
        filename="resume.txt"
    )

    job_text = (
        "Position: Full Stack Developer\n"
        "Required Skills: Python, Flask, Docker, Kubernetes, SQL"
    )

    comparison = job_rag.compare_job_with_resume(
        user_id=user_id,
        job_text=job_text,
        resume_skills=["python", "flask", "sql", "git", "rest api"]
    )

    assert comparison["success"] is True
    matched = [s.lower() for s in comparison["matched_skills"]]
    missing = [s.lower() for s in comparison["missing_skills"]]

    assert "python" in matched
    assert "flask" in matched
    assert "sql" in matched
    assert "docker" in missing
    assert "kubernetes" in missing
    assert len(comparison["learning_roadmap"]) >= 1

    # Cleanup
    VectorStoreService.get_instance().clear_user_collection(user_id, "resume")
    VectorStoreService.get_instance().clear_user_collection(user_id, "jobs")
