import os
import secrets
from datetime import timedelta

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from routes import auth_bp, github_bp, health_bp, resume_bp, rag_bp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "build"))
FRONTEND_STATIC_DIR = os.path.join(FRONTEND_BUILD_DIR, "static")

app = Flask(__name__, static_folder=FRONTEND_STATIC_DIR, static_url_path="/static")
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(os.getenv("RENDER")),
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
)
CORS(app)

# Register blueprints
app.register_blueprint(resume_bp, url_prefix="/api/resume")
app.register_blueprint(health_bp, url_prefix="/api")
app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(github_bp, url_prefix="/api/github")
app.register_blueprint(rag_bp, url_prefix="/api/rag")


def get_home_payload():
    return {
        "message": "AI Resume Analyzer",
        "homepage_title": "AI Resume Analyzer Smart ATS scoring, skill extraction, and interview preparation in one dashboard.",
        "status": "active",
        "version": "2.0",
        "features": [
            "Resume upload",
            "Live analysis dashboard",
            "Animated ATS score ring",
            "Skill match cards",
            "Missing keyword detector",
            "Job role prediction",
            "Career readiness score",
            "Prioritized action plan",
            "Application kit generator",
            "Detailed job keyword comparison",
            "Resume strength/weakness report",
            "Interview question generator",
            "Download analysis report as PDF",
            "Resume history page",
            "Dark/light mode",
            "RAG AI Assistant",
            "Semantic Resume Q&A",
            "Evidence-grounded Job Gap Analysis",
            "Career Knowledge Base",
        ],
        "dashboard_layout": {
            "top": "ATS Score Card | Skills Found | Missing Keywords | Suggested Role",
            "middle": "Resume Match Progress | Extracted Skills | Missing Skills",
            "bottom": "Strengths | Weaknesses | Interview Questions | Improvement Tips",
        },
        "endpoints": {
            "upload": "/api/resume/upload",
            "resume_metadata": "/api/resume/metadata",
            "job_match": "/api/resume/job-match",
            "login": "/api/auth/login",
            "profile": "/api/auth/profile/<username>",
            "subscribe": "/api/auth/subscribe",
            "github_login": "/api/github/login",
            "github_callback": "/api/github/callback",
            "github_connect": "/api/github/connect",
            "health": "/api/health",
            "rag_chat": "/api/rag/chat",
            "rag_ingest": "/api/rag/ingest",
            "rag_status": "/api/rag/status",
            "rag_sources": "/api/rag/sources",
            "rag_job_analysis": "/api/rag/job-analysis",
            "rag_interview_prep": "/api/rag/interview-preparation",
        },
    }


def serve_frontend(path="index.html"):
    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if not os.path.isfile(index_path):
        return jsonify({
            "error": "Frontend build not found.",
            "fix": "Run `npm run build` in the frontend directory, then restart Flask.",
        }), 503

    file_path = os.path.join(FRONTEND_BUILD_DIR, path)
    if path != "index.html" and os.path.isfile(file_path):
        return send_from_directory(FRONTEND_BUILD_DIR, path)

    return send_from_directory(FRONTEND_BUILD_DIR, "index.html")


@app.route("/")
def home():
    if request.args.get("format") == "json" or "application/json" in request.headers.get("Accept", ""):
        return jsonify(get_home_payload())

    return serve_frontend()


@app.route("/api/info")
def api_info():
    return jsonify(get_home_payload())


@app.route("/<path:path>")
def frontend_routes(path):
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found."}), 404

    return serve_frontend(path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
