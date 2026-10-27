"""
api.py

FastAPI wrapper for scholAR.

This file exposes the existing retrieval/RAG/question-generation engine as
JSON endpoints for a frontend. It deliberately does not duplicate business
logic; endpoints call the same functions used by the CLI tools.

Run from the repository root:
    uvicorn backend.src.api:app --reload

Open:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import sys
import json
import re
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Support both:
#   uvicorn backend.src.api:app --reload
# and:
#   cd backend/src; uvicorn api:app --reload
sys.path.insert(0, os.path.dirname(__file__))

from bm25_index import build_and_save_bm25_index  # noqa: E402
from chat_store import (  # noqa: E402
    add_message,
    create_session,
    delete_session,
    get_context_window,
    get_session,
    get_session_messages,
    list_sessions,
    update_session_title_from_query,
)
from chunking import chunk_pages  # noqa: E402
from contextual_query import rewrite_query_with_context  # noqa: E402
from embed_index import (  # noqa: E402
    DOCUMENT_METADATA_PATH,
    PDF_ROOT,
    compute_file_hash,
    embed_and_index_chunks,
    get_chroma_collection,
    load_document_metadata,
    make_document_id,
    relative_source_path,
)
from extract import extract_text  # noqa: E402
from generate_questions import generate_questions  # noqa: E402
from rag_answer import answer_question  # noqa: E402
from search import (  # noqa: E402
    DEFAULT_MODE,
    assess_retrieval_confidence,
    build_metadata_filters,
    search_chunks,
)


APP_NAME = "scholAR API"
APP_VERSION = "0.1.0"


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for academic hybrid search, cited RAG, and study question generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = os.path.join(PDF_ROOT, "uploads")


class MetadataFilters(BaseModel):
    subject: str | None = None
    doc_type: str | None = None
    source_type: str | None = None
    min_credibility: int | None = Field(default=None, ge=0, le=100)


RetrievalMode = Literal["vector", "hybrid", "hybrid_mmr"]


class SearchRequest(MetadataFilters):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source_file: str | None = None
    mode: RetrievalMode = DEFAULT_MODE


class AnswerRequest(MetadataFilters):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    source_file: str | None = None
    mode: RetrievalMode = DEFAULT_MODE
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1400, ge=100, le=4000)
    timeout_seconds: int | None = Field(default=None, ge=5, le=180)
    include_context: bool = False


class QuestionsRequest(MetadataFilters):
    topic: str = Field(..., min_length=1)
    question_type: Literal["short", "mcq"] = "short"
    count: int = Field(default=3, ge=1, le=10)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    top_k: int = Field(default=5, ge=1, le=10)
    source_file: str | None = None
    mode: RetrievalMode = DEFAULT_MODE
    temperature: float = Field(default=0.35, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1800, ge=100, le=5000)
    timeout_seconds: int | None = Field(default=None, ge=5, le=180)
    include_context: bool = False


class CreateSessionRequest(BaseModel):
    title: str | None = None


class ChatAnswerRequest(MetadataFilters):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    source_file: str | None = None
    mode: RetrievalMode = DEFAULT_MODE
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1400, ge=100, le=4000)
    timeout_seconds: int | None = Field(default=None, ge=5, le=180)
    context_messages: int = Field(default=5, ge=0, le=12)
    include_context: bool = False


def _safe_filename(filename: str) -> str:
    """
    Keep uploaded filenames readable but safe for local storage.
    """
    name = os.path.basename(filename or "uploaded.pdf")
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "uploaded"
    return f"{stem}{ext.lower() or '.pdf'}"


def _save_document_metadata(entry: dict) -> None:
    """
    Add or update a document metadata entry.

    document_metadata.json is our lightweight MVP metadata store. Later this
    naturally moves to Postgres/SQLite, but the contract stays the same.
    """
    data = load_document_metadata()
    documents = data.get("documents", [])
    documents = [
        document for document in documents
        if document.get("document_id") != entry["document_id"]
    ]
    documents.append(entry)

    os.makedirs(os.path.dirname(DOCUMENT_METADATA_PATH), exist_ok=True)
    with open(DOCUMENT_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump({"documents": documents}, file, indent=2)


def _metadata_filters_from_request(request: MetadataFilters) -> dict:
    return build_metadata_filters(
        subject=request.subject,
        doc_type=request.doc_type,
        source_type=request.source_type,
        min_credibility=request.min_credibility,
    )


def _preview(text: str, max_chars: int = 350) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _api_search_result(result: dict) -> dict:
    metadata = result.get("metadata", {})
    return {
        "rank": result.get("rank"),
        "id": result.get("id"),
        "distance": result.get("distance"),
        "scores": {
            "vector": result.get("vector_score"),
            "bm25": result.get("bm25_score"),
            "keyword": result.get("keyword_score"),
            "heading": result.get("heading_score"),
            "credibility": result.get("credibility_score_normalized"),
            "hybrid": result.get("hybrid_score"),
            "mmr": result.get("mmr_score"),
            "redundancy_penalty": result.get("mmr_diversity_penalty"),
        },
        "document": result.get("document"),
        "preview": _preview(result.get("document", "")),
        "metadata": metadata,
    }


def _strip_context(response: dict, include_context: bool) -> dict:
    if include_context:
        return response
    return {key: value for key, value in response.items() if key != "context"}


def _index_pdf(pdf_path: str, document_id: str, source_path: str, file_hash: str) -> dict:
    """
    Run the existing ingestion pipeline for one PDF.
    """
    extracted = extract_text(pdf_path)
    chunks = chunk_pages(extracted["pages"])

    for chunk in chunks:
        chunk["document_id"] = document_id
        chunk["source_path"] = source_path
        chunk["file_hash"] = file_hash

    summary = embed_and_index_chunks(chunks)
    bm25_index = build_and_save_bm25_index(get_chroma_collection())

    return {
        "filename": extracted["filename"],
        "page_count": extracted["page_count"],
        "chunks_created": len(chunks),
        "indexed": summary["indexed"],
        "skipped_not_embeddable": summary["skipped_not_embeddable"],
        "skipped_already_present": summary["skipped_already_present"],
        "bm25_chunk_count": bm25_index["document_count"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/stats")
def stats() -> dict:
    collection = get_chroma_collection()
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
    pdf_count = 0
    if os.path.isdir(pdf_dir):
        pdf_count = sum(
            1
            for _, _, files in os.walk(pdf_dir)
            for filename in files
            if filename.lower().endswith(".pdf")
        )

    return {
        "collection": "resource_chunks",
        "chunk_count": collection.count(),
        "pdf_count": pdf_count,
    }


@app.post("/search")
def search_endpoint(request: SearchRequest) -> dict:
    try:
        results = search_chunks(
            query_text=request.query,
            top_k=request.top_k,
            source_file=request.source_file,
            mode=request.mode,
            metadata_filters=_metadata_filters_from_request(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return {
        "query": request.query,
        "mode": request.mode,
        "top_k": request.top_k,
        "retrieval_confidence": assess_retrieval_confidence(results, mode=request.mode),
        "results": [_api_search_result(result) for result in results],
    }


@app.post("/answer")
def answer_endpoint(request: AnswerRequest) -> dict:
    try:
        response = answer_question(
            request.query,
            top_k=request.top_k,
            source_file=request.source_file,
            metadata_filters=_metadata_filters_from_request(request),
            mode=request.mode,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            timeout_seconds=request.timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc

    return _strip_context(response, include_context=request.include_context)


@app.post("/questions")
def questions_endpoint(request: QuestionsRequest) -> dict:
    try:
        response = generate_questions(
            request.topic,
            question_type=request.question_type,
            count=request.count,
            difficulty=request.difficulty,
            top_k=request.top_k,
            source_file=request.source_file,
            metadata_filters=_metadata_filters_from_request(request),
            mode=request.mode,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            timeout_seconds=request.timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {exc}") from exc

    return _strip_context(response, include_context=request.include_context)


@app.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    subject: str = Form("unknown"),
    course_code: str = Form("unknown"),
    doc_type: str = Form("unknown"),
    source_type: str = Form("student_uploaded"),
    verified_status: Literal["unknown", "verified", "unverified"] = Form("unknown"),
    semester: int = Form(0),
    year: int = Form(0),
    exam_type: str = Form("unknown"),
    institution: str = Form("unknown"),
    uploaded_by: str = Form("user"),
) -> dict:
    """
    Upload one PDF and immediately add it to Chroma + BM25.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    safe_name = _safe_filename(file.filename)
    temp_path = os.path.join(UPLOAD_DIR, f"tmp_{safe_name}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

        with open(temp_path, "wb") as output:
            output.write(contents)

        file_hash = compute_file_hash(temp_path)
        stem, ext = os.path.splitext(safe_name)
        final_filename = f"{stem}_{file_hash[:10]}{ext}"
        final_path = os.path.join(UPLOAD_DIR, final_filename)

        if os.path.exists(final_path):
            os.remove(temp_path)
        else:
            os.replace(temp_path, final_path)

        source_path = relative_source_path(final_path)
        document_id = make_document_id(source_path, file_hash)

        metadata_entry = {
            "document_id": document_id,
            "source_path": source_path,
            "original_filename": file.filename,
            "file_hash": file_hash,
            "subject": subject or "unknown",
            "course_code": course_code or "unknown",
            "doc_type": doc_type or "unknown",
            "source_type": source_type or "student_uploaded",
            "verified_status": verified_status or "unknown",
            "semester": semester or 0,
            "year": year or 0,
            "exam_type": exam_type or "unknown",
            "institution": institution or "unknown",
            "uploaded_by": uploaded_by or "user",
        }
        _save_document_metadata(metadata_entry)

        indexing = _index_pdf(final_path, document_id, source_path, file_hash)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload/indexing failed: {exc}") from exc

    return {
        "status": "indexed",
        "document_id": document_id,
        "source_path": source_path,
        "file_hash": file_hash,
        "metadata": metadata_entry,
        "indexing": indexing,
    }


@app.post("/sessions")
def create_session_endpoint(request: CreateSessionRequest) -> dict:
    try:
        return create_session(title=request.title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not create session: {exc}") from exc


@app.get("/sessions")
def list_sessions_endpoint() -> dict:
    try:
        return {"sessions": list_sessions()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not list sessions: {exc}") from exc


@app.get("/sessions/{session_id}")
def get_session_endpoint(session_id: str) -> dict:
    try:
        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {
            "session": session,
            "messages": get_session_messages(session_id),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid session id: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load session: {exc}") from exc


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str) -> dict:
    try:
        deleted = delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid session id: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete session: {exc}") from exc


@app.post("/sessions/{session_id}/answer")
def chat_answer_endpoint(session_id: str, request: ChatAnswerRequest) -> dict:
    """
    Context-aware RAG answer inside a persistent chat session.

    The current user query is rewritten into a standalone retrieval query using
    the previous N messages, then normal RAG runs over that retrieval query.
    """
    try:
        session = get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        history = get_context_window(
            session_id,
            limit=request.context_messages,
        )
        rewrite = rewrite_query_with_context(
            request.query,
            history,
            timeout_seconds=request.timeout_seconds,
        )

        add_message(
            session_id=session_id,
            role="user",
            content=request.query,
        )
        update_session_title_from_query(session_id, request.query)

        response = answer_question(
            request.query,
            retrieval_query=rewrite["query"],
            conversation_context=history,
            top_k=request.top_k,
            source_file=request.source_file,
            metadata_filters=_metadata_filters_from_request(request),
            mode=request.mode,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            timeout_seconds=request.timeout_seconds,
        )
        response["query_rewrite"] = rewrite

        assistant_message = add_message(
            session_id=session_id,
            role="assistant",
            content=response["answer"],
            sources=response.get("sources"),
            output_credibility=response.get("output_credibility"),
            retrieval_confidence=response.get("retrieval_confidence"),
        )
        response["message"] = assistant_message

        return _strip_context(response, include_context=request.include_context)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat answer failed: {exc}") from exc
