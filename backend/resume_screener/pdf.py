from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 4 * 1024 * 1024  # Vercel request bodies are capped at 4.5 MB


class PdfError(ValueError):
    pass


def pdf_to_text(data: bytes) -> str:
    if not data.startswith(b"%PDF"):
        raise PdfError("file is not a PDF")
    if len(data) > MAX_PDF_BYTES:
        raise PdfError("PDF is larger than 4 MB")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as e:
        raise PdfError("could not read this PDF") from e
    text = "\n".join(p for p in pages if p)
    if not text.strip():
        raise PdfError("no text found in the PDF (scanned PDFs are not supported)")
    return text
