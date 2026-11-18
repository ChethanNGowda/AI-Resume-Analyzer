"""
Regression tests for existing ResumeNova functionality.
Ensures ATS scoring, job matching, authentication, health checks, and metadata remain intact.
"""

import io
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "ResumeNova API"


def test_api_info_home_payload(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "active"
    assert "upload" in data["endpoints"]
    assert "rag_chat" in data["endpoints"]


def test_auth_login_existing_users(client):
    # Test valid user login
    res = client.post("/api/auth/login", json={"username": "user", "password": "user123"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["username"] == "user"
    assert data["role"] == "user"

    # Test invalid password
    res_bad = client.post("/api/auth/login", json={"username": "user", "password": "wrongpassword"})
    assert res_bad.status_code == 401

    # Test missing fields
    res_empty = client.post("/api/auth/login", json={})
    assert res_empty.status_code == 400


def test_resume_metadata_endpoint(client):
    res = client.get("/api/resume/metadata")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "mock_interview" in data["features"]
    assert "resume_analyzer" in data["features"]


def test_job_match_existing_logic(client):
    payload = {
        "resume_skills": ["python", "flask", "react", "sql"],
        "job_description": "We are seeking a Python developer with SQL, Docker, and AWS knowledge."
    }
    res = client.post("/api/resume/job-match", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "matched_skills" in data
    assert "missing_skills" in data
    assert "match_score" in data
    assert "fit_level" in data
    assert "python" in data["matched_skills"]
    assert "sql" in data["matched_skills"]
    assert "docker" in data["missing_skills"]
    assert "aws" in data["missing_skills"]


def test_upload_missing_file_error(client):
    res = client.post("/api/resume/upload", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
