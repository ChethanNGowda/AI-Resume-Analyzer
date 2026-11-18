"""
Ollama LLM Client and Grounded Generation Service.
Interfaces with local Ollama instance (default model: qwen3:8b) with graceful fallback and strict evidence grounding.
"""

import logging
import json
from typing import Dict, List, Optional
import requests

from services.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class LLMService:
    """
    Client for local Ollama LLM inference.
    Enforces strict context grounding, citations, and handles model unavailability.
    """

    def __init__(self):
        self.base_url = RAGConfig.OLLAMA_BASE_URL
        self.model_name = RAGConfig.RAG_LLM_MODEL
        self.timeout = RAGConfig.OLLAMA_TIMEOUT

    def check_health(self) -> Dict:
        """
        Check if Ollama is reachable and check if target model is installed.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=4)
            if resp.status_code == 200:
                models_data = resp.json().get("models", [])
                installed_models = [m.get("name", "") for m in models_data]
                has_target_model = any(
                    self.model_name in name or name.startswith(self.model_name.split(":")[0])
                    for name in installed_models
                )
                return {
                    "available": True,
                    "ollama_url": self.base_url,
                    "target_model": self.model_name,
                    "target_model_installed": has_target_model,
                    "installed_models": installed_models,
                }
            return {
                "available": False,
                "error": f"Ollama returned HTTP status {resp.status_code}",
                "target_model": self.model_name,
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"Cannot connect to Ollama at {self.base_url}: {e}",
                "target_model": self.model_name,
            }

    def generate_grounded_response(
        self,
        prompt: str,
        context_chunks: List[Dict],
        system_role: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate a response grounded in retrieved context chunks.
        If Ollama is not available or returns an error, generates a structured grounded fallback.
        """
        citations = []
        context_text_parts = []

        for idx, chunk in enumerate(context_chunks):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Document")
            sec_hint = meta.get("section_hint", "General")
            page_num = meta.get("page_number", 1)
            chunk_text = chunk.get("text", "").strip()

            citation_label = f"[{doc_name} | {sec_hint} (p.{page_num})]"
            citations.append({
                "citation_id": idx + 1,
                "source": doc_name,
                "section": sec_hint,
                "page": page_num,
                "similarity": chunk.get("similarity", 0.0),
                "label": citation_label,
            })
            context_text_parts.append(f"--- SOURCE {idx + 1}: {citation_label} ---\n{chunk_text}")

        formatted_context = "\n\n".join(context_text_parts) if context_text_parts else "No matching documents found."

        system_prompt = system_role or (
            "You are ResumeNova's AI Career Assistant and Resume Coach.\n"
            "CRITICAL RULES:\n"
            "1. Answer strictly and solely using the provided retrieved context sources.\n"
            "2. If the answer cannot be determined from the provided context, state clearly and honestly: "
            "'This information is not present in your uploaded resume/documents.'\n"
            "3. DO NOT invent, fabricate, or assume any qualifications, marks, projects, companies, skills, or dates.\n"
            "4. Cite the source reference (e.g. [Document.pdf | Section]) whenever stating a specific fact.\n"
            "5. Maintain an encouraging, professional, and actionable tone."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"=== RETRIEVED EVIDENCE SOURCES ===\n"
            f"{formatted_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{prompt}\n\n"
            f"=== GROUNDED RESPONSE ==="
        )

        # Attempt Ollama inference
        health = self.check_health()
        if health.get("available"):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temperature for factual grounding
                        "top_p": 0.9,
                    }
                }
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result.get("response", "").strip()
                    if answer:
                        return {
                            "response": answer,
                            "citations": citations,
                            "model_used": self.model_name,
                            "grounded": True,
                            "llm_online": True,
                        }
            except Exception as e:
                logger.warning(f"Ollama generation failed ({e}). Using deterministic grounded fallback.")

        # Fallback grounded synthesis without calling LLM
        fallback_answer = self._synthesize_grounded_fallback(prompt, context_chunks, citations)
        return {
            "response": fallback_answer,
            "citations": citations,
            "model_used": "resumenova-grounded-fallback",
            "grounded": True,
            "llm_online": False,
        }

    def _synthesize_grounded_fallback(
        self,
        prompt: str,
        context_chunks: List[Dict],
        citations: List[Dict]
    ) -> str:
        """
        Deterministic, rule-guided synthesis when local Ollama is offline or warming up.
        Guarantees that responses remain grounded in actual retrieved chunks without hallucinations.
        """
        if not context_chunks:
            return (
                "I searched your uploaded documents, but could not find relevant information "
                "addressing this question. Please upload a resume or job description covering this topic."
            )

        lower_prompt = prompt.lower()
        top_chunk = context_chunks[0]
        top_text = top_chunk.get("text", "")
        doc_name = top_chunk.get("metadata", {}).get("document_name", "Resume")

        if any(w in lower_prompt for w in ["skill", "technical", "stack"]):
            # Extract bullet lines from chunks
            points = [line.strip() for line in top_text.splitlines() if line.strip() and len(line.strip()) > 3]
            selected = "\n".join(f"- {p}" for p in points[:6])
            return (
                f"Based on your uploaded {doc_name}, the following skills and qualifications were retrieved:\n\n"
                f"{selected}\n\n"
                f"[Source: {citations[0]['label'] if citations else doc_name}]"
            )

        if any(w in lower_prompt for w in ["education", "degree", "university", "college", "gpa"]):
            edu_lines = [
                line.strip() for line in top_text.splitlines()
                if any(k in line.lower() for k in ["bachelor", "master", "degree", "b.tech", "university", "college", "gpa", "school", "education"])
            ]
            if edu_lines:
                return (
                    f"According to your {doc_name}, your educational details are:\n\n"
                    + "\n".join(f"- {l}" for l in edu_lines[:4])
                    + f"\n\n[Source: {citations[0]['label'] if citations else doc_name}]"
                )
            return (
                f"Here is the relevant section retrieved from {doc_name}:\n\n"
                f"{top_text[:400]}...\n\n"
                f"[Source: {citations[0]['label'] if citations else doc_name}]"
            )

        if any(w in lower_prompt for w in ["project", "projects"]):
            return (
                f"Here are the project details retrieved from your {doc_name}:\n\n"
                f"{top_text[:500]}\n\n"
                f"[Source: {citations[0]['label'] if citations else doc_name}]"
            )

        # General excerpt grounded answer
        summary_lines = [l.strip() for l in top_text.splitlines() if l.strip()][:5]
        body = "\n".join(f"- {l}" for l in summary_lines)
        return (
            f"Here is what was found in your {doc_name} regarding your inquiry:\n\n"
            f"{body}\n\n"
            f"Source: {citations[0]['label'] if citations else doc_name}"
        )
