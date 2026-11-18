"""
Document chunker and prompt injection sanitizer for ResumeNova RAG pipeline.
Splits text into semantically cohesive chunks while preserving structural metadata.
"""

import hashlib
import re
from typing import Dict, List, Optional
from services.rag.config import RAGConfig

# Common resume and technical document section header patterns
SECTION_HEADER_PATTERNS = [
    r"(?i)^(?:technical\s+)?skills(?:\s+and\s+competencies)?\b",
    r"(?i)^(?:work\s+)?experience\b",
    r"(?i)^employment\s+history\b",
    r"(?i)^education(?:\s+and\s+qualifications)?\b",
    r"(?i)^projects?\b",
    r"(?i)^certifications?(?:\s+and\s+licenses)?\b",
    r"(?i)^summary(?:\s+of\s+qualifications)?\b",
    r"(?i)^professional\s+summary\b",
    r"(?i)^objective\b",
    r"(?i)^awards?(?:\s+and\s+honors)?\b",
    r"(?i)^publications?\b",
    r"(?i)^languages?\b",
    r"(?i)^responsibilities\b",
    r"(?i)^requirements\b",
    r"(?i)^qualifications\b",
]

# Patterns representing prompt injection attempts in uploaded documents
PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
    r"(?i)disregard\s+(?:all\s+)?prior\s+prompts",
    r"(?i)system\s*:\s*you\s+are",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)\[INST\]",
    r"(?i)\[/INST\]",
    r"(?i)###\s*(?:system|instruction|response)",
    r"(?i)new\s+instruction\s*:\s*",
    r"(?i)you\s+must\s+say\s+that\s+the\s+candidate\s+is\s+perfect",
    r"(?i)give\s+(?:this\s+candidate\s+)?a\s+score\s+of\s+100",
]


def sanitize_prompt_injection(text: str) -> str:
    """
    Neutralize potential prompt injection attempts inside uploaded document text.
    Replaces deceptive directives with neutral disarmed text.
    """
    if not text:
        return ""

    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(
            pattern,
            lambda m: f"[FILTERED_DIRECTIVE: {m.group(0)[:20]}...]",
            sanitized
        )

    # Disarm XML/HTML-like tags that could attempt to trick LLM template engines
    sanitized = re.sub(r"<(script|iframe|prompt|instruction)[\s>].*?</\1>", "[BLOCKED_TAG]", sanitized, flags=re.I | re.S)
    return sanitized


def detect_section_hint(line: str) -> Optional[str]:
    """Check if a line matches a known resume or job description section header."""
    clean_line = line.strip().rstrip(":-#* ")
    if len(clean_line) > 60:
        return None

    for pattern in SECTION_HEADER_PATTERNS:
        if re.search(pattern, clean_line):
            return clean_line.title()
    return None


class DocumentChunker:
    """
    Splits document text into overlapping chunks with rich metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or RAGConfig.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or RAGConfig.RAG_CHUNK_OVERLAP

    def chunk_document(
        self,
        parsed_doc: Dict,
        source_id: str,
        user_id: str,
        document_type: str = "resume"
    ) -> List[Dict]:
        """
        Takes output from DocumentParser and chunks it into metadata-enriched segments.
        Returns a list of chunk dicts ready for embedding and ChromaDB storage.
        """
        full_text = parsed_doc.get("text", "")
        filename = parsed_doc.get("filename", "unknown_doc.txt")
        pages = parsed_doc.get("pages", [])

        # Sanitize against prompt injections
        safe_text = sanitize_prompt_injection(full_text)

        # If pages are present, chunk page-aware; otherwise chunk full text
        chunks: List[Dict] = []
        if pages and len(pages) > 1:
            chunk_idx = 0
            for page_info in pages:
                p_text = sanitize_prompt_injection(page_info.get("text", ""))
                p_num = page_info.get("page_number", 1)
                page_chunks = self._split_text_to_chunks(p_text)
                for pc in page_chunks:
                    chunk_id = f"{source_id}_chunk_{chunk_idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "user_id": user_id,
                        "document_name": filename,
                        "document_type": document_type,
                        "chunk_index": chunk_idx,
                        "page_number": p_num,
                        "section_hint": pc.get("section_hint", "General"),
                        "text": pc["text"],
                        "char_count": len(pc["text"]),
                    })
                    chunk_idx += 1
        else:
            raw_chunks = self._split_text_to_chunks(safe_text)
            for chunk_idx, rc in enumerate(raw_chunks):
                chunk_id = f"{source_id}_chunk_{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "user_id": user_id,
                    "document_name": filename,
                    "document_type": document_type,
                    "chunk_index": chunk_idx,
                    "page_number": 1,
                    "section_hint": rc.get("section_hint", "General"),
                    "text": rc["text"],
                    "char_count": len(rc["text"]),
                })

        # Set total_chunks across all
        total = len(chunks)
        for c in chunks:
            c["total_chunks"] = total

        return chunks

    def _split_text_to_chunks(self, text: str) -> List[Dict]:
        """
        Splits text recursively by paragraphs -> lines -> sentences -> words.
        Attaches section hint if a section header is encountered.
        """
        if not text:
            return []

        # Track active section
        paragraphs = text.split("\n\n")
        chunks = []
        current_section = "Overview"
        current_buffer = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph starts with a section header
            first_line = para.split("\n")[0]
            detected_sec = detect_section_hint(first_line)
            if detected_sec:
                current_section = detected_sec

            para_len = len(para)

            # If adding this paragraph exceeds chunk_size and we already have content
            if current_len + para_len > self.chunk_size and current_buffer:
                chunk_text = "\n\n".join(current_buffer).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "section_hint": current_section,
                    })

                # Handle overlap: retain end of current buffer
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else ""
                current_buffer = [overlap_text, para] if overlap_text else [para]
                current_len = sum(len(x) for x in current_buffer)
            elif para_len > self.chunk_size:
                # Single paragraph exceeds chunk size -> split by lines/sentences
                sub_parts = self._split_large_paragraph(para)
                for sp in sub_parts:
                    if current_buffer and current_len + len(sp) > self.chunk_size:
                        chunk_text = "\n\n".join(current_buffer).strip()
                        if chunk_text:
                            chunks.append({
                                "text": chunk_text,
                                "section_hint": current_section,
                            })
                        current_buffer = [sp]
                        current_len = len(sp)
                    else:
                        current_buffer.append(sp)
                        current_len += len(sp)
            else:
                current_buffer.append(para)
                current_len += para_len

        if current_buffer:
            chunk_text = "\n\n".join(current_buffer).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "section_hint": current_section,
                })

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into sentence- or word-based chunks."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        parts = []
        curr = []
        curr_len = 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if curr_len + len(s) > self.chunk_size and curr:
                parts.append(" ".join(curr))
                curr = [s]
                curr_len = len(s)
            else:
                curr.append(s)
                curr_len += len(s)

        if curr:
            parts.append(" ".join(curr))

        # If any single sentence is still larger than chunk_size, hard slice by words
        final_parts = []
        for p in parts:
            if len(p) <= self.chunk_size:
                final_parts.append(p)
            else:
                words = p.split()
                w_buff = []
                w_len = 0
                for w in words:
                    if w_len + len(w) + 1 > self.chunk_size and w_buff:
                        final_parts.append(" ".join(w_buff))
                        w_buff = [w]
                        w_len = len(w)
                    else:
                        w_buff.append(w)
                        w_len += len(w) + 1
                if w_buff:
                    final_parts.append(" ".join(w_buff))

        return final_parts
