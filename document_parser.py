"""Document parsing utilities for VibeVox.

Supports PDF, EPUB, and TXT file formats.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional


def parse_txt(content: bytes) -> str:
    """Parse plain text file."""
    return content.decode("utf-8", errors="replace")


def parse_pdf(content: bytes) -> str:
    """Parse PDF file and extract text."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")

    text_parts = []
    with io.BytesIO(content) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    
    return "\n\n".join(text_parts)


def parse_epub(content: bytes) -> str:
    """Parse EPUB file and extract text."""
    try:
        import ebooklib
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "ebooklib and beautifulsoup4 are required for EPUB parsing. "
            "Install with: pip install ebooklib beautifulsoup4"
        )

    text_parts = []
    with io.BytesIO(content) as epub_file:
        book = ebooklib.epub.read_epub(epub_file)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    text_parts.append(text)

    return "\n\n".join(text_parts)


def parse_document(
    content: bytes,
    filename: str,
) -> str:
    """Parse document based on file extension.
    
    Args:
        content: Raw file content bytes
        filename: Name of the file (used to determine format)
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file format is not supported
    """
    path = Path(filename)
    extension = path.suffix.lower()
    
    if extension in {".txt", ".text"}:
        return parse_txt(content)
    elif extension == ".pdf":
        return parse_pdf(content)
    elif extension in {".epub"}:
        return parse_epub(content)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats: .txt, .pdf, .epub"
        )


def detect_encoding(content: bytes) -> str:
    """Attempt to detect text encoding.
    
    Args:
        content: Raw file content bytes
        
    Returns:
        Detected encoding name
    """
    try:
        import chardet
        result = chardet.detect(content)
        return result.get("encoding", "utf-8") or "utf-8"
    except ImportError:
        return "utf-8"
