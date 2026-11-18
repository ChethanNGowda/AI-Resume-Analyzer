# ResumeNova RAG API Reference

This document outlines the REST API endpoints provided by ResumeNova's Retrieval-Augmented Generation (RAG) module.

All RAG routes are mounted under the `/api/rag` prefix.

---

## Authentication & Headers

ResumeNova associates documents with user accounts:
- Authenticated requests should include the header:
  `X-User-Id: <username>` (or `X-Username: <username>`).
- If omitted, the request falls back to an isolated guest session based on client IP.

---

## 1. System Status

### `GET /api/rag/status`
Checks the operational status of the RAG subsystems (Ollama, ChromaDB, SentenceTransformers).

#### Response `200 OK`
```json
{
  "success": true,
  "rag_enabled": true,
  "llm": {
    "available": true,
    "ollama_url": "http://localhost:11434",
    "target_model": "qwen3:8b",
    "target_model_installed": true,
    "installed_models": ["qwen3:8b:latest"]
  },
  "vector_database": {
    "available": true,
    "engine": "ChromaDB",
    "storage_path": "backend/data/chroma",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "collections_count": 3
  },
  "config": {
    "model": "qwen3:8b",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "chunk_size": 800,
    "chunk_overlap": 120
  }
}
```

---

## 2. Ingest Document

### `POST /api/rag/ingest`
Ingests a document (PDF, DOCX, TXT, MD) into the user's isolated collection.

#### Headers
- `Content-Type: multipart/form-data` or `application/json`
- `X-User-Id: <username>`

#### Parameters (Multipart Form)
- `file`: The document binary file.
- `type`: Target collection type: `"resume"` (default) or `"job_description"`.

#### Parameters (JSON Body)
```json
{
  "text": "Full text of resume or job description...",
  "filename": "my_resume.txt",
  "type": "resume"
}
```

#### Response `201 Created`
```json
{
  "success": true,
  "data": {
    "source_id": "resume_489a41606a2de2d2",
    "document_name": "jane_resume.pdf",
    "status": "indexed",
    "chunk_count": 8,
    "word_count": 420,
    "char_count": 2840,
    "text_preview": "Jane Doe Software Engineer..."
  }
}
```

---

## 3. Conversational AI Chat

### `POST /api/rag/chat`
Answers questions strictly grounded in retrieved documents with source citations.

#### Request Body
```json
{
  "message": "What technical skills do I have in my resume?",
  "context": "auto"
}
```
*Note: `context` can be `"auto"` (hybrid), `"resume"` (user resume only), `"job"` (job descriptions only), or `"career"` (curated career guides).*

#### Response `200 OK`
```json
{
  "success": true,
  "data": {
    "response": "According to your uploaded resume [jane_resume.pdf | Skills (p.1)], you possess expertise in Python, Flask, SQL, React, and Docker.",
    "citations": [
      {
        "citation_id": 1,
        "source": "jane_resume.pdf",
        "section": "Skills",
        "page": 1,
        "similarity": 0.8921,
        "label": "[jane_resume.pdf | Skills (p.1)]"
      }
    ],
    "model_used": "qwen3:8b",
    "grounded": true,
    "llm_online": true,
    "context_used": "resume"
  },
  "user_id": "jane_doe"
}
```

---

## 4. List User Ingested Sources

### `GET /api/rag/sources`
Lists all documents ingested for the current authenticated user.

#### Response `200 OK`
```json
{
  "success": true,
  "user_id": "jane_doe",
  "resumes": [
    {
      "source_id": "resume_489a41606a2de2d2",
      "document_name": "jane_resume.pdf",
      "document_type": "resume",
      "chunk_count": 8,
      "user_id": "jane_doe"
    }
  ],
  "jobs": [],
  "total": 1
}
```

---

## 5. Delete Document Source

### `DELETE /api/rag/sources/<source_id>`
Deletes a specific document and its vector embeddings from the user's isolated collection.

#### Response `200 OK`
```json
{
  "success": true,
  "message": "Source 'resume_489a41606a2de2d2' removed successfully."
}
```

---

## 6. Job Description Semantic Gap Analysis

### `POST /api/rag/job-analysis`
Semantically compares a target job description against the user's stored resume chunks.

#### Request Body
```json
{
  "job_description": "We are seeking a Senior Backend Engineer with Python, Flask, Docker, and Kubernetes experience.",
  "resume_skills": ["python", "flask", "sql"]
}
```

#### Response `200 OK`
```json
{
  "success": true,
  "data": {
    "job_title": "Senior Backend Engineer",
    "work_arrangement": "Not specified",
    "experience_required": "Not specified",
    "match_score": 60,
    "matched_skills": ["python", "flask"],
    "missing_skills": ["docker", "kubernetes"],
    "evidence": [
      {
        "source": "jane_resume.pdf",
        "section": "Technical Skills",
        "page": 1,
        "snippet": "Python, Flask, SQL, REST APIs..."
      }
    ],
    "learning_roadmap": [
      "Study Docker fundamentals and complete one hands-on project demonstrating containerization.",
      "Study Kubernetes fundamentals and complete one hands-on project demonstrating practical implementation."
    ],
    "resume_improvement_suggestions": [
      "Add verifiable evidence for 'docker' in your project bullet points.",
      "Add verifiable evidence for 'kubernetes' in your project bullet points."
    ],
    "has_resume_context": true
  }
}
```

---

## 7. Role-Specific Interview Preparation

### `POST /api/rag/interview-preparation`
Generates targeted interview questions and suggested answers grounded in the user's actual projects.

#### Request Body
```json
{
  "target_role": "Backend Engineer"
}
```

#### Response `200 OK`
```json
{
  "success": true,
  "target_role": "Backend Engineer",
  "data": {
    "response": "Here are 5 targeted interview questions generated from your resume projects:\n1. Walk me through how you built your REST API in Flask...",
    "citations": [
      {
        "citation_id": 1,
        "source": "jane_resume.pdf",
        "section": "Projects",
        "page": 1
      }
    ],
    "grounded": true
  },
  "has_resume_evidence": true
}
```

---

## 8. Clear Conversation

### `POST /api/rag/clear-conversation`
Clears chat turn history for the user's active session.

#### Response `200 OK`
```json
{
  "success": true,
  "message": "Conversation cleared."
}
```

---

## 9. List Career Topics

### `GET /api/rag/career-topics`
Lists curated reference guides available in the career knowledge base.

#### Response `200 OK`
```json
{
  "success": true,
  "topics": [
    { "id": "data_analytics", "title": "Data Analytics", "filename": "data_analytics.md" },
    { "id": "python_engineering", "title": "Python Engineering", "filename": "python_engineering.md" },
    { "id": "sql_database_mastery", "title": "Sql Database Mastery", "filename": "sql_database_mastery.md" },
    { "id": "technical_interviews", "title": "Technical Interviews", "filename": "technical_interviews.md" },
    { "id": "ats_optimization", "title": "Ats Optimization", "filename": "ats_optimization.md" }
  ]
}
```
