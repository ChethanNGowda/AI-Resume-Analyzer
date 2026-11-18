"""
Tests for document parsing (PDF, DOCX, TXT, MD) and text cleaning.
"""

import io
import pytest
from services.rag.document_parser import (
    DocumentParser,
    DocumentParseError,
    clean_text,
    sanitize_filename,
    compute_content_hash,
)


def test_clean_text():
    raw = "Hello   world!\r\n\r\n\n\nThis is\t a test.\u00A0"
    cleaned = clean_text(raw)
    assert "Hello world!" in cleaned
    assert "This is a test." in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("my resume (1) [final].pdf") == "my resume _1_ _final_.pdf"
    assert sanitize_filename("") == "unnamed_document.txt"


def test_compute_content_hash():
    data1 = b"Sample resume content 1"
    data2 = b"Sample resume content 1"
    data3 = b"Sample resume content 2"
    assert compute_content_hash(data1) == compute_content_hash(data2)
    assert compute_content_hash(data1) != compute_content_hash(data3)


def test_parse_txt_document():
    content = "John Doe\nSoftware Engineer\nSkills: Python, Flask, SQL"
    parsed = DocumentParser.parse_file(content.encode("utf-8"), filename="resume.txt")
    assert parsed["filename"] == "resume.txt"
    assert parsed["file_type"] == "txt"
    assert "John Doe" in parsed["text"]
    assert parsed["word_count"] > 5
    assert parsed["char_count"] == len(parsed["text"])


def test_parse_markdown_document():
    content = "# Senior Developer\n## Experience\n- Built scalable REST APIs using Flask"
    parsed = DocumentParser.parse_file(content.encode("utf-8"), filename="skills.md")
    assert parsed["file_type"] == "md"
    assert "Built scalable REST APIs" in parsed["text"]


def test_parse_docx_document():
    import docx
    doc = docx.Document()
    doc.add_heading("Alice Smith", level=1)
    doc.add_paragraph("Experienced Machine Learning Engineer with expertise in PyTorch and ChromaDB.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skill"
    table.rows[0].cells[1].text = "Python"

    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)

    parsed = DocumentParser.parse_file(stream.getvalue(), filename="alice_resume.docx")
    assert "Alice Smith" in parsed["text"]
    assert "PyTorch and ChromaDB" in parsed["text"]
    assert "Python" in parsed["text"]
    assert parsed["file_type"] == "docx"


def test_empty_document_raises_error():
    with pytest.raises(DocumentParseError, match="Document is empty"):
        DocumentParser.parse_file(b"", filename="empty.txt")


def test_unsupported_document_type():
    with pytest.raises(DocumentParseError):
        DocumentParser.parse_file(b"\x00\x01\x02\x03\x04\x05", filename="binary.exe")
