"""
chat_store.py

Persistent chat sessions/messages for scholAR.

This layer deliberately uses plain SQL. It is easy to explain, easy to migrate,
and maps cleanly to PostgreSQL/Neon.
"""

from __future__ import annotations

import json
from uuid import UUID

from db import get_connection


DEFAULT_CONTEXT_MESSAGES = 5


def init_chat_schema() -> None:
    """
    Create chat tables if they do not exist.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL DEFAULT 'New chat',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json JSONB,
                    credibility_json JSONB,
                    retrieval_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages (session_id, created_at);
                """
            )


def _validate_uuid(value: str) -> str:
    return str(UUID(value))


def _row_to_session(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def _row_to_message(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "session_id": str(row["session_id"]),
        "role": row["role"],
        "content": row["content"],
        "sources": row.get("sources_json") or [],
        "output_credibility": row.get("credibility_json"),
        "retrieval_confidence": row.get("retrieval_json"),
        "created_at": row["created_at"].isoformat(),
    }


def create_session(title: str | None = None) -> dict:
    init_chat_schema()
    clean_title = (title or "New chat").strip() or "New chat"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (title)
                VALUES (%s)
                RETURNING id, title, created_at, updated_at;
                """,
                (clean_title,),
            )
            return _row_to_session(cur.fetchone())


def list_sessions(limit: int = 50) -> list[dict]:
    init_chat_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return [_row_to_session(row) for row in cur.fetchall()]


def get_session(session_id: str) -> dict | None:
    init_chat_schema()
    session_id = _validate_uuid(session_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE id = %s;
                """,
                (session_id,),
            )
            row = cur.fetchone()
            return _row_to_session(row) if row else None


def add_message(
    *,
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    output_credibility: dict | None = None,
    retrieval_confidence: dict | None = None,
) -> dict:
    init_chat_schema()
    session_id = _validate_uuid(session_id)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'.")
    if not content.strip():
        raise ValueError("message content cannot be empty.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (
                    session_id,
                    role,
                    content,
                    sources_json,
                    credibility_json,
                    retrieval_json
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                RETURNING id, session_id, role, content, sources_json,
                          credibility_json, retrieval_json, created_at;
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(sources) if sources is not None else None,
                    json.dumps(output_credibility) if output_credibility is not None else None,
                    json.dumps(retrieval_confidence) if retrieval_confidence is not None else None,
                ),
            )
            message = _row_to_message(cur.fetchone())
            cur.execute(
                """
                UPDATE chat_sessions
                SET updated_at = NOW()
                WHERE id = %s;
                """,
                (session_id,),
            )
            return message


def get_session_messages(session_id: str) -> list[dict]:
    init_chat_schema()
    session_id = _validate_uuid(session_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, role, content, sources_json,
                       credibility_json, retrieval_json, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC;
                """,
                (session_id,),
            )
            return [_row_to_message(row) for row in cur.fetchall()]


def delete_session(session_id: str) -> bool:
    """
    Delete a chat session.

    chat_messages are removed automatically by ON DELETE CASCADE.
    Returns True if a session was deleted.
    """
    init_chat_schema()
    session_id = _validate_uuid(session_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE id = %s
                RETURNING id;
                """,
                (session_id,),
            )
            return cur.fetchone() is not None


def get_context_window(
    session_id: str,
    limit: int = DEFAULT_CONTEXT_MESSAGES,
) -> list[dict]:
    init_chat_schema()
    session_id = _validate_uuid(session_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (session_id, limit),
            )
            rows = list(reversed(cur.fetchall()))
            return [
                {"role": row["role"], "content": row["content"]}
                for row in rows
            ]


def update_session_title_from_query(session_id: str, query: str) -> None:
    """
    Give new sessions a useful title based on the first user query.
    """
    session_id = _validate_uuid(session_id)
    clean = " ".join(query.split())
    if not clean:
        return
    title = clean[:60]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET title = %s, updated_at = NOW()
                WHERE id = %s AND title = 'New chat';
                """,
                (title, session_id),
            )
