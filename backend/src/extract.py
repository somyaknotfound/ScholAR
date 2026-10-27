"""
extract.py

Phase 1:
- Digital PDFs only
- Uses pymupdf4llm
- Extracts page-by-page markdown
- Preserves page numbers
- Attaches basic metadata

Future:
- OCR fallback
- PPT support
- Image support
"""

import os
import sys
import io
import re
import fitz
import pymupdf4llm

PICTURE_PLACEHOLDER = "==> picture omitted <=="
PICTURE_OMITTED_PATTERN = re.compile(r"picture\b.*?\bomitted", re.IGNORECASE)
LOW_TEXT_WORD_THRESHOLD = 25
IMAGE_HEAVY_WORD_THRESHOLD = 80
MIN_OCR_WORDS_TO_ACCEPT = 10
OCR_ZOOM = 2.5


def get_page_count(pdf_path: str) -> int:
    """
    Returns total number of pages in the PDF.
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()
    return page_count


def _word_count(text: str) -> int:
    """
    Count real text words, ignoring pymupdf4llm image placeholders.
    """
    cleaned = PICTURE_OMITTED_PATTERN.sub(" ", text.replace(PICTURE_PLACEHOLDER, " "))
    return len(cleaned.split())


def _should_try_ocr(text: str) -> bool:
    """
    Decide whether native extraction looks too weak and OCR is worth trying.

    OCR is treated as a fallback, not a replacement for normal PDF text.
    Native text is usually cleaner than OCR, so we only try OCR for low-text
    or image-placeholder-heavy pages.
    """
    word_count = _word_count(text)
    has_picture_placeholder = (
        PICTURE_PLACEHOLDER in text
        or PICTURE_OMITTED_PATTERN.search(text) is not None
    )

    return (
        word_count < LOW_TEXT_WORD_THRESHOLD
        or (has_picture_placeholder and word_count < IMAGE_HEAVY_WORD_THRESHOLD)
    )


def _ocr_page(pdf_path: str, page_index: int) -> str:
    """
    Render one PDF page and run Tesseract OCR on it.

    pytesseract is optional at runtime. If it or the Tesseract executable is
    unavailable, OCR simply returns an empty string and normal extraction is
    preserved.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""
    finally:
        if doc is not None:
            doc.close()


def _use_ocr_if_better(pdf_path: str, page_index: int, native_text: str) -> str:
    """
    Keep native extraction unless OCR clearly recovers more usable text.
    """
    if not _should_try_ocr(native_text):
        return native_text

    ocr_text = _ocr_page(pdf_path, page_index)
    native_words = _word_count(native_text)
    ocr_words = _word_count(ocr_text)

    if ocr_words >= MIN_OCR_WORDS_TO_ACCEPT and ocr_words > native_words:
        return ocr_text

    return native_text


def extract_pages(pdf_path: str, filename: str) -> list:
    """
    Extract page-wise markdown text.

    Returns:
    [
        {"page_number": 1, "source_file": "os_notes.pdf", "text": "..."},
        ...
    ]
    """

    page_data = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True
    )

    pages = []

    for i, page in enumerate(page_data):
        text = page.get("text", "").strip()
        text = _use_ocr_if_better(pdf_path, i, text)

        pages.append(
            {
                "page_number": i + 1,
                "source_file": filename,
                "text": text
            }
        )

    return pages


def extract_text(pdf_path: str) -> dict:
    """
    Main extraction entry point.

    Returns:
    {
        filename,
        file_path,
        file_type,
        page_count,
        pages
    }
    """

    filename = os.path.basename(pdf_path)

    file_type = (
        os.path.splitext(filename)[1]
        .lower()
        .lstrip(".")
    )

    result = {
        "filename": filename,
        "file_path": os.path.abspath(pdf_path),
        "file_type": file_type,
        "page_count": 0,
        "pages": []
    }

    try:
        result["page_count"] = get_page_count(pdf_path)

    except Exception as e:
        print(
            f"[WARN] Could not read page count "
            f"for {pdf_path}: {e}"
        )

    try:
        result["pages"] = extract_pages(
            pdf_path,
            filename
        )

    except Exception as e:
        print(
            f"[ERROR] Extraction failed "
            f"for {pdf_path}: {e}"
        )

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python extract.py <path_to_pdf>"
        )
        sys.exit(1)

    output = extract_text(sys.argv[1])

    print("\n========================")
    print("EXTRACTION SUMMARY")
    print("========================")

    print(f"File      : {output['filename']}")
    print(f"Type      : {output['file_type']}")
    print(f"Pages     : {output['page_count']}")
    print(f"Path      : {output['file_path']}")

    print(
        f"Pages Extracted : "
        f"{len(output['pages'])}"
    )

    if output["pages"]:
        print("\n--- FIRST PAGE PREVIEW ---")
        print(output["pages"][0]["text"][:500])
