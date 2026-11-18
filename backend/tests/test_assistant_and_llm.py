"""
Tests for Conversational AI Assistant and LLM grounding.
Uses mocked Ollama HTTP responses so the test suite does not require a running Ollama daemon.
"""

from unittest.mock import MagicMock, patch
import pytest
from services.rag.assistant_service import AssistantService
from services.rag.llm_service import LLMService


@pytest.fixture
def assistant():
    return AssistantService()


def test_intent_detection(assistant):
    assert assistant._detect_intent("What technical skills are in my resume?") == "resume"
    assert assistant._detect_intent("Does my background fit this job description?") == "job"
    assert assistant._detect_intent("How do I answer behavioral interview questions?") == "career"
    assert assistant._detect_intent("Tell me about software engineering placement prep") == "career"


def test_mocked_ollama_generation():
    llm = LLMService()

    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "response": "Based on your resume [Jane_Resume.pdf | Education], you hold a Bachelor of Technology degree."
    }

    mock_tags_resp = MagicMock()
    mock_tags_resp.status_code = 200
    mock_tags_resp.json.return_value = {"models": [{"name": "qwen3:8b"}]}

    with patch("requests.get", return_value=mock_tags_resp), \
         patch("requests.post", return_value=mock_ollama_resp):

        context_chunks = [
            {
                "chunk_id": "c1",
                "text": "Bachelor of Technology in Computer Engineering, graduated 2024.",
                "similarity": 0.88,
                "metadata": {"document_name": "Jane_Resume.pdf", "section_hint": "Education", "page_number": 1}
            }
        ]

        result = llm.generate_grounded_response("What is my degree?", context_chunks)
        assert result["llm_online"] is True
        assert "Bachelor of Technology" in result["response"]
        assert len(result["citations"]) == 1
        assert result["citations"][0]["source"] == "Jane_Resume.pdf"


def test_ollama_offline_fallback():
    llm = LLMService()

    # Simulate connection error to Ollama
    with patch("requests.get", side_effect=Exception("Connection refused")), \
         patch("requests.post", side_effect=Exception("Connection refused")):

        context_chunks = [
            {
                "chunk_id": "c1",
                "text": "Technical Skills: Python, SQL, Docker, React",
                "similarity": 0.82,
                "metadata": {"document_name": "Resume.pdf", "section_hint": "Skills", "page_number": 1}
            }
        ]

        result = llm.generate_grounded_response("What skills do I have?", context_chunks)
        assert result["llm_online"] is False
        assert result["grounded"] is True
        assert "Python" in result["response"]


def test_conversation_history_and_clearing(assistant):
    user_id = "test_user_convo"
    assistant.clear_conversation(user_id)

    assistant.chat(user_id=user_id, message="Hello assistant")
    history = assistant.get_conversation_history(user_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    assistant.clear_conversation(user_id)
    assert len(assistant.get_conversation_history(user_id)) == 0
