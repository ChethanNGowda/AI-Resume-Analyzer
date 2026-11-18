"""
Integration tests for RAG API blueprint routes.
"""

import io
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_rag_status_endpoint(client):
    res = client.get("/api/rag/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "llm" in data
    assert "vector_database" in data
    assert "config" in data


def test_rag_career_topics(client):
    res = client.get("/api/rag/career-topics")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["topics"]) >= 10


def test_rag_ingest_and_sources(client):
    headers = {"X-User-Id": "integration_tester"}

    # Ingest text payload
    res = client.post(
        "/api/rag/ingest",
        json={"text": "Skills: Python, Flask, PyTest, SQL\nProjects: Built RAG chatbot.", "filename": "test.txt"},
        headers=headers,
    )
    assert res.status_code == 201
    ingest_data = res.get_json()
    assert ingest_data["success"] is True

    # List sources
    res_sources = client.get("/api/rag/sources", headers=headers)
    assert res_sources.status_code == 200
    sources_data = res_sources.get_json()
    assert sources_data["user_id"] == "integration_tester"
    assert len(sources_data["resumes"]) >= 1

    source_id = sources_data["resumes"][0]["source_id"]

    # Chat with ingested resume
    chat_res = client.post(
        "/api/rag/chat",
        json={"message": "What skills do I have?", "context": "resume"},
        headers=headers,
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.get_json()
    assert chat_data["success"] is True
    assert "Python" in chat_data["data"]["response"] or "skills" in chat_data["data"]["response"].lower()

    # Delete source
    del_res = client.delete(f"/api/rag/sources/{source_id}", headers=headers)
    assert del_res.status_code == 200


def test_rag_job_analysis(client):
    headers = {"X-User-Id": "job_tester"}
    job_payload = {
        "job_description": "We are seeking a Python developer with experience in Flask, Docker, and PostgreSQL.",
        "resume_skills": ["python", "flask"],
    }
    res = client.post("/api/rag/job-analysis", json=job_payload, headers=headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "python" in [s.lower() for s in data["data"]["matched_skills"]]
    assert "docker" in [s.lower() for s in data["data"]["missing_skills"]]


def test_rag_interview_preparation(client):
    headers = {"X-User-Id": "interview_tester"}
    res = client.post(
        "/api/rag/interview-preparation",
        json={"target_role": "Backend Engineer"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["data"]["response"]) > 20


def test_rag_clear_conversation(client):
    headers = {"X-User-Id": "clear_tester"}
    res = client.post("/api/rag/clear-conversation", json={}, headers=headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
