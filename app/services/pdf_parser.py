"""PDF and DOCX text extraction service."""

from __future__ import annotations

import asyncio
import io
from fastapi import UploadFile
import pdfplumber
from docx import Document

# Magic bytes for file type validation (not just extension)
_PDF_MAGIC = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"  # DOCX is a ZIP archive


async def extract_text(file: UploadFile) -> str:
    """Extract text from a PDF or DOCX upload (async-safe).

    Validates file type via magic bytes, not just extension.
    Runs sync PDF/DOCX parsing in a thread to avoid blocking the event loop.
    """
    filename = (file.filename or "").lower()
    file.file.seek(0)
    content = file.file.read()

    if not content:
        raise ValueError(f"{filename}: file is empty.")

    # Validate by magic bytes first, then fallback to extension
    is_pdf = content[:4] == _PDF_MAGIC
    is_docx = content[:4] == _DOCX_MAGIC and filename.endswith(".docx")

    if is_pdf or (not is_docx and filename.endswith(".pdf")):
        if not is_pdf:
            raise ValueError(
                f"{filename}: file extension is .pdf but content is not a valid PDF."
            )
        return await asyncio.to_thread(_extract_pdf, content)

    if is_docx or filename.endswith(".docx"):
        if not is_docx:
            raise ValueError(
                f"{filename}: file extension is .docx but content is not a valid DOCX."
            )
        return await asyncio.to_thread(_extract_docx, content)

    raise ValueError(f"Unsupported file type: {filename}. Use PDF or DOCX.")


def _extract_pdf(content: bytes) -> str:
    """Extract text from raw PDF bytes using pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _extract_docx(content: bytes) -> str:
    """Extract text from raw DOCX bytes."""
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
