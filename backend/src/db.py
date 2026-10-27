"""
db.py

Small PostgreSQL connection layer for scholAR chat history.

The app uses ChromaDB for vectors and BM25 for lexical retrieval. PostgreSQL is
used for relational app data: chat sessions and chat messages.
"""

from __future__ import annotations

import os

from llm_provider import load_env_file


load_env_file()

DATABASE_URL = os.getenv("DATABASE_URL")


def require_database_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Neon/PostgreSQL connection "
            "string to .env, e.g. DATABASE_URL=postgresql://..."
        )
    return DATABASE_URL


def get_connection():
    """
    Return a psycopg connection.

    Importing psycopg inside the function keeps the rest of the app importable
    until chat endpoints are actually used.
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Run: pip install 'psycopg[binary]'"
        ) from exc

    return psycopg.connect(require_database_url(), row_factory=dict_row)
