"""
Document parser supporting PDF, DOCX, TXT, and Markdown files.
Performs text extraction, cleaning, metadata generation, and security sanitization.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx
except ImportError:
    docx = None


class DocumentParseError(Exception):
    """Raised when document parsing fails due to invalid, corrupt, or unsupported data."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and script injection.
    """
    if not filename:
        return "unnamed_document.txt"

    # Strip directory components
    name = Path(filename).name
    # Remove null bytes and control chars
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # Remove dangerous characters like .. or / or \
    name = re.sub(r"[^\w\s\.-]", "_", name).strip()
    return name if name else "document.txt"


def clean_text(raw_text: str) -> str:
    """
    Clean unnecessary whitespace, repeated empty lines, and non-printable characters.
    """
    if not raw_text:
        return ""

    # Normalize carriage returns
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove null bytes and non-printable control characters (except newline, tab)
    text = re.sub(r"[^\x20-\x7E\n\t\u00A0-\uFFFF]", " ", text)
    # Replace non-breaking spaces with standard space
    text = text.replace("\u00A0", " ")
    # Replace tabs with spaces
    text = text.replace("\t", "    ")
    # Collapse multiple spaces on the same line into one
    text = re.sub(r"[ ]{2,}", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_content_hash(content_bytes: bytes) -> str:
    """Compute sha256 hash of document bytes for duplicate detection."""
    return hashlib.sha256(content_bytes).hexdigest()


class DocumentParser:
    """
    Parses and extracts clean text along with metadata from multiple document types.
    """

    @classmethod
    def parse_file(
        cls,
        file_input: Union[str, Path, io.BytesIO, bytes],
        filename: Optional[str] = None
    ) -> Dict:
        """
        Parses a document file and returns cleaned text with comprehensive metadata.
        Returns:
            {
                "text": str,
                "filename": str,
                "file_type": str,
                "content_hash": str,
                "word_count": int,
                "char_count": int,
                "pages": List[Dict],  # if available
                "page_count": int,
            }
        """
        raw_bytes: bytes
        derived_filename: str = "document.txt"

        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.is_file():
                raise DocumentParseError(f"File not found: {file_input}")
            derived_filename = path.name
            with open(path, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
            if filename:
                derived_filename = filename
        elif hasattr(file_input, "read"):
            # Handles Flask FileStorage or BytesIO
            if hasattr(file_input, "filename") and file_input.filename:
                derived_filename = file_input.filename
            elif filename:
                derived_filename = filename

            # Save stream position and read bytes
            curr_pos = file_input.tell() if hasattr(file_input, "tell") else 0
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(curr_pos)
        else:
            raise DocumentParseError("Unsupported file input type.")

        if not raw_bytes or len(raw_bytes.strip()) == 0:
            raise DocumentParseError("Document is empty (0 bytes).")

        clean_name = sanitize_filename(filename or derived_filename)
        ext = Path(clean_name).suffix.lower()
        content_hash = compute_content_hash(raw_bytes)

        pages_data: List[Dict] = []
        raw_extracted_text = ""

        if ext == ".pdf":
            raw_extracted_text, pages_data = cls._parse_pdf(raw_bytes)
        elif ext in (".docx", ".doc"):
            raw_extracted_text = cls._parse_docx(raw_bytes)
        elif ext in (".txt", ".md", ".markdown"):
            raw_extracted_text = cls._parse_text(raw_bytes)
        else:
            # Fallback text parse attempt
            try:
                raw_extracted_text = cls._parse_text(raw_bytes)
            except Exception:
                raise DocumentParseError(f"Unsupported document format: {ext}")

        cleaned_text = clean_text(raw_extracted_text)
        if not cleaned_text:
            raise DocumentParseError("Could not extract any readable text from document.")

        words = cleaned_text.split()
        return {
            "text": cleaned_text,
            "filename": clean_name,
            "file_type": ext.lstrip(".").lower() or "text",
            "content_hash": content_hash,
            "word_count": len(words),
            "char_count": len(cleaned_text),
            "pages": pages_data,
            "page_count": len(pages_data) if pages_data else 1,
        }

    @classmethod
    def _parse_pdf(cls, pdf_bytes: bytes) -> Tuple[str, List[Dict]]:
        """Extract text page-by-page from PDF bytes using pdfplumber."""
        if not pdfplumber:
            raise DocumentParseError("pdfplumber is required for PDF parsing.")

        extracted_text_parts: List[str] = []
        pages_info: List[Dict] = []

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    raise DocumentParseError("PDF contains zero pages.")

                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as pe:
                        page_text = ""

                    cleaned_page_text = clean_text(page_text)
                    if cleaned_page_text:
                        extracted_text_parts.append(cleaned_page_text)
                        pages_info.append({
                            "page_number": page_num,
                            "text": cleaned_page_text,
                            "char_count": len(cleaned_page_text),
                        })

        except Exception as e:
            if isinstance(e, DocumentParseError):
                raise
            err_msg = str(e).lower()
            if "password" in err_msg or "encrypt" in err_msg:
                raise DocumentParseError("PDF document is encrypted or password-protected.")
            raise DocumentParseError(f"Failed to parse PDF document: {e}")

        full_text = "\n\n".join(extracted_text_parts)
        return full_text, pages_info

    @classmethod
    def _parse_docx(cls, docx_bytes: bytes) -> str:
        """Extract text from DOCX paragraphs and tables."""
        if not docx:
            raise DocumentParseError("python-docx is required for DOCX parsing.")

        try:
            doc = docx.Document(io.BytesIO(docx_bytes))
            text_parts: List[str] = []

            for paragraph in doc.paragraphs:
                p_text = paragraph.text.strip()
                if p_text:
                    text_parts.append(p_text)

            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        text_parts.append(" | ".join(row_cells))

            return "\n\n".join(text_parts)
        except Exception as e:
            raise DocumentParseError(f"Failed to parse DOCX document: {e}")

    @classmethod
    def _parse_text(cls, raw_bytes: bytes) -> str:
        """Extract text from plain text or markdown with encoding auto-detection."""
        encodings = ["utf-8", "latin-1", "utf-16", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                return raw_bytes.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback with replacement
        return raw_bytes.decode("utf-8", errors="replace")
