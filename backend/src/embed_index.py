"""
embed_index.py

Takes chunks from chunking.py, generates embeddings, and stores them in
ChromaDB along with their metadata.

CHUNK IDENTITY: this is the layer that decides a chunk's permanent ID -
not chunking.py. The ID is a hash of (source_file + page range +
normalized chunk_text), not a position-based index. This means:
  - re-running chunking.py with a different chunk_size, or after fixing
    something in extract.py, produces a DIFFERENT id for genuinely
    different content, and the SAME id for unchanged content
  - the "skip if already indexed" check below is therefore actually
    correct - same id really does mean same content, always
  - chunking.py's own sequential chunk_id (e.g. "deadlocks_pdf_0003") is
    only used as a position label internally; it is not what gets stored
    as the Chroma id

SKIPPED CHUNKS: chunks with should_embed=False (near-empty, picture-only
placeholders) are not embedded or stored at all in Phase 1. This is a
deliberate, reversible scope choice - once Phase 2 adds real image/diagram
understanding, those chunks will naturally get should_embed=True from
chunking.py and will be picked up on the next run, with no redesign needed
here.
"""

import hashlib
import os

# Keep Hugging Face / Transformers output quiet during normal search runs.
# The model is still downloaded on first use if it is not already cached.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(
    __file__), "..", "data", "db", "chroma_store")
DOCUMENT_METADATA_PATH = os.path.join(os.path.dirname(
    __file__), "..", "data", "document_metadata.json")
PDF_ROOT = os.path.abspath(os.path.join(os.path.dirname(
    __file__), "..", "data", "pdfs"))
COLLECTION_NAME = "resource_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly
CHUNKING_VERSION = "page_markdown_v1"

DEFAULT_DOCUMENT_METADATA = {
    "document_id": "unknown",
    "source_path": "unknown",
    "file_hash": "unknown",
    "original_filename": "unknown",
    "subject": "unknown",
    "course_code": "unknown",
    "doc_type": "unknown",
    "source_type": "unknown",
    "verified_status": "unknown",
    "semester": 0,
    "year": 0,
    "exam_type": "unknown",
    "institution": "unknown",
    "uploaded_by": "unknown",
}

_model = None  # lazy-loaded singleton, so repeated calls don't reload the model


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            # Prefer the local cache so normal search does not make a Hugging
            # Face Hub request every time a new Python process starts.
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                local_files_only=True,
            )
        except Exception:
            # First run on a new machine: allow sentence-transformers to
            # download/cache the model, then future runs should load locally.
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_chroma_collection():
    """
    Delete and recreate the Chroma collection, and remove the stored BM25
    lexical index.

    Used by --rebuild so the vector index exactly matches the current PDFs,
    extraction logic, chunking logic, embedding model, lexical index, and
    metadata schema.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing Chroma collection: {COLLECTION_NAME}")
    except Exception as exc:
        print(f"No existing Chroma collection deleted: {exc}")

    from bm25_index import delete_bm25_index, BM25_INDEX_PATH

    delete_bm25_index()
    print(f"Deleted existing BM25 lexical index if present: {BM25_INDEX_PATH}")

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _normalize_for_id(text: str) -> str:
    """
    Normalize whitespace before hashing so harmless formatting changes do not
    create a new vector id.
    """
    return " ".join(text.split())


def _normalize_path(path: str) -> str:
    """
    Store paths with forward slashes so metadata works consistently across
    Windows/macOS/Linux.
    """
    return path.replace("\\", "/")


def relative_source_path(pdf_path: str) -> str:
    """
    Return the PDF path relative to backend/data/pdfs when possible.
    This is useful for display and metadata matching.
    """
    absolute_pdf_path = os.path.abspath(pdf_path)
    try:
        if os.path.commonpath([absolute_pdf_path, PDF_ROOT]) == PDF_ROOT:
            return _normalize_path(os.path.relpath(absolute_pdf_path, PDF_ROOT))
    except ValueError:
        pass
    return _normalize_path(os.path.basename(pdf_path))


def compute_file_hash(file_path: str) -> str:
    """
    SHA-256 fingerprint of the actual PDF bytes.

    This avoids relying on filenames for identity and lets the system detect
    exact duplicate uploads later.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_document_id(source_path: str, file_hash: str) -> str:
    """
    Deterministic document-record id.

    The id includes both source path and file hash:
      - same filename in different folders will not collide
      - same content in different upload records can still be distinguished
      - file_hash remains available to detect exact duplicate content
    """
    raw = f"{_normalize_path(source_path)}::{file_hash}".encode("utf-8")
    return f"doc_{hashlib.sha256(raw).hexdigest()[:24]}"


def make_stable_chunk_id(chunk: dict) -> str:
    """
    Content-based id with page provenance.

    Same source file + same page range + same normalized text -> same id,
    regardless of chunk position. Including the page range prevents repeated
    text on different pages from collapsing into one Chroma record.
    """
    document_key = (
        chunk.get("document_id")
        or chunk.get("source_path")
        or chunk.get("source_file")
        or "unknown_document"
    )
    start_page = chunk.get("start_page") or 0
    end_page = chunk.get("end_page") or 0
    normalized_text = _normalize_for_id(chunk.get("chunk_text", ""))

    raw = f"{document_key}::{start_page}-{end_page}::{normalized_text}".encode("utf-8")
    # short enough to stay readable
    return hashlib.sha256(raw).hexdigest()[:24]


def load_document_metadata() -> dict:
    """
    Load per-document metadata used for filtering and credibility scoring.

    This JSON file is the CLI/dev equivalent of a future upload form where the
    user selects subject, document type, source type, verification status, etc.
    Missing files or missing document entries fall back to explicit "unknown"
    values instead of guessing.
    """
    import json

    if not os.path.exists(DOCUMENT_METADATA_PATH):
        return {}

    with open(DOCUMENT_METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _document_metadata_entries(all_document_metadata: dict) -> list[dict]:
    """
    Support the newer {"documents": [...]} format and the older
    {"filename.pdf": {...}} format.
    """
    if isinstance(all_document_metadata.get("documents"), list):
        return all_document_metadata["documents"]

    entries = []
    for filename, metadata in all_document_metadata.items():
        if isinstance(metadata, dict):
            entry = metadata.copy()
            entry.setdefault("original_filename", filename)
            entry.setdefault("source_path", filename)
            entries.append(entry)
    return entries


def _metadata_for_document(chunk: dict, all_document_metadata: dict) -> dict:
    """
    Return metadata for a document.

    Matching priority:
      1. document_id
      2. file_hash
      3. source_path
      4. original filename / source_file fallback
    """
    entries = _document_metadata_entries(all_document_metadata)
    document_id = chunk.get("document_id")
    file_hash = chunk.get("file_hash")
    source_path = _normalize_path(chunk.get("source_path") or "")
    source_file = (chunk.get("source_file") or "").lower()

    raw_metadata = {}
    for entry in entries:
        entry_source_path = _normalize_path(entry.get("source_path", ""))
        entry_filename = (
            entry.get("original_filename")
            or entry.get("source_file")
            or os.path.basename(entry_source_path)
            or ""
        ).lower()

        if document_id and entry.get("document_id") == document_id:
            raw_metadata = entry
            break
        if file_hash and entry.get("file_hash") == file_hash:
            raw_metadata = entry
            break
        if source_path and entry_source_path and entry_source_path.lower() == source_path.lower():
            raw_metadata = entry
            break
        if source_file and entry_filename == source_file:
            raw_metadata = entry
            break

    metadata = DEFAULT_DOCUMENT_METADATA.copy()
    metadata.update(raw_metadata or {})
    return metadata


def _credibility_score(document_metadata: dict) -> tuple[int, str]:
    """
    Simple, explainable credibility score.

    The score is based on uploader/document metadata, not AI guesswork. Unknown
    fields are allowed but receive little or no boost.
    """
    score = 0

    if document_metadata.get("verified_status") == "verified":
        score += 35
    elif document_metadata.get("verified_status") == "unverified":
        score -= 10

    source_type = document_metadata.get("source_type")
    if source_type in {"faculty_notes", "official_college"}:
        score += 25
    elif source_type in {"official_publisher", "conference", "journal"}:
        score += 25
    elif source_type == "textbook":
        score += 22
    elif source_type == "previous_year_paper":
        score += 20
    elif source_type in {"arxiv", "research_repository"}:
        score += 15
    elif source_type == "web_resource":
        score += 10
    elif source_type == "student_uploaded":
        score += 5

    doc_type = document_metadata.get("doc_type")
    if doc_type == "question_paper":
        score += 20
    elif doc_type in {"textbook", "lecture_notes"}:
        score += 15
    elif doc_type == "research_paper":
        score += 25
    elif doc_type in {"assignment", "lab_manual", "syllabus"}:
        score += 10
    elif doc_type == "student_notes":
        score += 5

    if document_metadata.get("subject") != "unknown":
        score += 10
    if document_metadata.get("course_code") != "unknown":
        score += 5
    if document_metadata.get("year", 0):
        score += 5

    score = max(0, min(100, score))
    if score >= 75:
        label = "high"
    elif score >= 45:
        label = "medium"
    else:
        label = "low"

    return score, label


def _build_metadata(chunk: dict, document_metadata: dict | None = None) -> dict:
    """
    Chroma metadata values must be str / int / float / bool - no lists or
    None. Lists are joined into strings; None becomes "unknown" or 0 so
    every field is always present and filterable later.
    """
    document_metadata = document_metadata or DEFAULT_DOCUMENT_METADATA
    credibility_score, credibility_label = _credibility_score(document_metadata)

    return {
        "document_id": chunk.get("document_id") or document_metadata.get("document_id", "unknown"),
        "source_file": chunk.get("source_file") or "unknown_source",
        "source_path": chunk.get("source_path") or document_metadata.get("source_path", "unknown"),
        "file_hash": chunk.get("file_hash") or document_metadata.get("file_hash", "unknown"),
        "original_filename": document_metadata.get(
            "original_filename",
            chunk.get("source_file") or "unknown_source",
        ),
        "start_page": chunk.get("start_page") or 0,
        "end_page": chunk.get("end_page") or 0,
        "pages": ",".join(str(p) for p in chunk.get("pages", [])),
        "section_heading": chunk.get("section_heading") or "unknown",
        "parent_headings": ",".join(chunk.get("parent_headings") or []),
        "image_count": chunk.get("image_count") or 0,
        "text_word_count": chunk.get("text_word_count") or 0,
        "merged_across_pages": bool(chunk.get("merged_across_pages")),
        "chunk_index": chunk.get("chunk_index") or 0,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunking_version": CHUNKING_VERSION,
        "usage_count": 0,  # incremented later, at search time
        "subject": document_metadata.get("subject", "unknown"),
        "course_code": document_metadata.get("course_code", "unknown"),
        "doc_type": document_metadata.get("doc_type", "unknown"),
        "source_type": document_metadata.get("source_type", "unknown"),
        "verified_status": document_metadata.get("verified_status", "unknown"),
        "semester": document_metadata.get("semester") or 0,
        "year": document_metadata.get("year") or 0,
        "exam_type": document_metadata.get("exam_type", "unknown"),
        "institution": document_metadata.get("institution", "unknown"),
        "uploaded_by": document_metadata.get("uploaded_by", "unknown"),
        "credibility_score": credibility_score,
        "credibility_label": credibility_label,
    }


def embed_and_index_chunks(chunks: list, batch_size: int = 32) -> dict:
    """
    Embeds and stores chunks in ChromaDB.

    chunks: list of chunk dicts from chunking.chunk_pages()
    Returns a summary dict: {indexed, skipped_not_embeddable, skipped_already_present}
    """
    summary = {"indexed": 0, "skipped_not_embeddable": 0,
               "skipped_already_present": 0}

    embeddable = [c for c in chunks if c.get("should_embed")]
    summary["skipped_not_embeddable"] = len(chunks) - len(embeddable)

    if not embeddable:
        return summary

    collection = get_chroma_collection()

    # compute stable ids first, so we can check which ones already exist
    candidates = []
    for chunk in embeddable:
        stable_id = make_stable_chunk_id(chunk)
        candidates.append((stable_id, chunk))

    existing_ids = set()
    try:
        existing = collection.get(ids=[c[0] for c in candidates])
        existing_ids = set(existing["ids"])
    except Exception as exc:
        print(f"Warning: could not check existing Chroma ids: {exc}")

    to_index = [(cid, chunk)
                for cid, chunk in candidates if cid not in existing_ids]
    summary["skipped_already_present"] = len(candidates) - len(to_index)

    if not to_index:
        return summary

    model = get_model()
    all_document_metadata = load_document_metadata()

    for batch_start in range(0, len(to_index), batch_size):
        batch = to_index[batch_start: batch_start + batch_size]
        batch_ids = [cid for cid, _ in batch]
        batch_texts = [chunk["chunk_text"] for _, chunk in batch]
        batch_metadatas = []
        for _, chunk in batch:
            document_metadata = _metadata_for_document(chunk, all_document_metadata)
            batch_metadatas.append(_build_metadata(chunk, document_metadata))

        embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadatas,
        )
        summary["indexed"] += len(batch)

    return summary


def collect_pdf_paths(input_paths: list[str]) -> list[str]:
    """
    Accept a mix of PDF files and directories.

    Directories are scanned recursively for .pdf files so the normal project
    path backend/data/pdfs can be indexed in one command.
    """
    pdf_paths = []

    for input_path in input_paths:
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for filename in files:
                    if filename.lower().endswith(".pdf"):
                        pdf_paths.append(os.path.join(root, filename))
        elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
            pdf_paths.append(input_path)
        else:
            print(f"Warning: skipping non-PDF path: {input_path}")

    return sorted(set(pdf_paths))


def main():
    import argparse
    import sys

    sys.path.insert(0, ".")
    from extract import extract_text
    from chunking import chunk_pages

    parser = argparse.ArgumentParser(
        description="Embed PDF chunks and store them in the local Chroma index."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more PDF files or directories containing PDFs.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to embed per batch. Default: 32.",
    )
    args = parser.parse_args()

    if args.rebuild:
        reset_chroma_collection()

    pdf_paths = collect_pdf_paths(args.paths)
    if not pdf_paths:
        print("No PDF files found to index.")
        sys.exit(1)

    print(f"PDF files to index: {len(pdf_paths)}")

    for pdf_path in pdf_paths:
        output = extract_text(pdf_path)
        chunks = chunk_pages(output["pages"])
        source_path = relative_source_path(pdf_path)
        file_hash = compute_file_hash(pdf_path)
        document_id = make_document_id(source_path, file_hash)

        for chunk in chunks:
            chunk["document_id"] = document_id
            chunk["source_path"] = source_path
            chunk["file_hash"] = file_hash

        print(
            f"\n{output['filename']}: {len(chunks)} chunks produced by chunking.py")
        print(f"  document_id: {document_id}")
        print(f"  source_path: {source_path}")
        summary = embed_and_index_chunks(chunks, batch_size=args.batch_size)
        print(
            f"  indexed: {summary['indexed']} | "
            f"skipped (should_embed=False): {summary['skipped_not_embeddable']} | "
            f"skipped (already in Chroma): {summary['skipped_already_present']}"
        )

    collection = get_chroma_collection()
    print(
        f"\nTotal chunks now in Chroma collection '{COLLECTION_NAME}': {collection.count()}")

    from bm25_index import build_and_save_bm25_index, BM25_INDEX_PATH

    bm25_index = build_and_save_bm25_index(collection)
    print(
        f"BM25 lexical index saved to {BM25_INDEX_PATH} "
        f"({bm25_index['document_count']} chunks)")


if __name__ == "__main__":
    main()
