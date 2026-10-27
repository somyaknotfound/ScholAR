"""
contextual_query.py

Rewrite follow-up chat questions into standalone retrieval queries.

Example:
  previous: "explain deadlock prevention"
  current:  "give me MCQs on that"
  rewrite:  "generate MCQs about deadlock prevention in operating systems"

The rewritten query is used for retrieval. The original query is still shown to
the answer-generation LLM so the response matches what the user actually asked.
"""

from __future__ import annotations

import re

from bm25_index import tokenize
from llm_provider import LLMProviderError, generate_with_fallback


SYSTEM_PROMPT = """
You rewrite student follow-up questions into standalone retrieval queries.

Rules:
- Use the previous conversation only to resolve references like "that",
  "this topic", "above", "it", or "same".
- Preserve the user's intent.
- Do not answer the question.
- Return only one rewritten query as plain text.
- If the current query is already standalone, return it unchanged.
- If the current query asks for questions/MCQs/quiz "on it/that", explicitly
  include the topic from the previous conversation.
""".strip()


VAGUE_MARKERS = {"that", "this", "it", "above", "same", "topic"}
GENERIC_REWRITE_TERMS = {
    "question",
    "questions",
    "practice",
    "mcq",
    "mcqs",
    "quiz",
    "answer",
    "answers",
    "generate",
    "give",
    "some",
}
INVALID_REWRITES = {".*", "*", ".", "?", "query", "search"}


def _format_history(messages: list[dict], max_chars: int = 1800) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "unknown")
        content = " ".join((message.get("content") or "").split())
        if not content:
            continue
        lines.append(f"{role}: {content}")

    history = "\n".join(lines)
    if len(history) <= max_chars:
        return history
    return history[-max_chars:]


def simple_contextual_query(current_query: str, history: list[dict]) -> str:
    """
    Cheap fallback when the LLM rewriter fails.

    It combines the last user question with the current query. This is not as
    smart as an LLM rewrite, but it often fixes vague queries like "that".
    """
    previous_user_queries = [
        message["content"]
        for message in history
        if message.get("role") == "user" and message.get("content")
    ]
    if not previous_user_queries:
        return current_query

    current_lower = current_query.lower()
    if not any(marker in current_lower.split() for marker in VAGUE_MARKERS):
        return current_query

    return f"{previous_user_queries[-1]} {current_query}"


def _last_user_query(history: list[dict]) -> str:
    previous_user_queries = [
        message["content"]
        for message in history
        if message.get("role") == "user" and message.get("content")
    ]
    return previous_user_queries[-1] if previous_user_queries else ""


def _needs_history_anchor(current_query: str) -> bool:
    raw_terms = set(re.findall(r"[a-z0-9]+", current_query.lower()))
    return bool(raw_terms & VAGUE_MARKERS)


def _dropped_history_topic(
    *,
    current_query: str,
    rewritten_query: str,
    history: list[dict],
) -> bool:
    """
    Detect unsafe LLM rewrites that remove the topic.

    Example bad rewrite:
      history:   "flow diffusion"
      current:   "generate a question on it"
      rewritten: "What are some practice questions"

    The rewritten query has no meaningful token from the last user topic, so
    retrieval becomes generic. In that case, use the deterministic fallback.
    """
    if not _needs_history_anchor(current_query):
        return False

    previous_query = _last_user_query(history)
    if not previous_query:
        return False

    previous_terms = set(tokenize(previous_query)) - GENERIC_REWRITE_TERMS
    rewritten_terms = set(tokenize(rewritten_query))

    if not previous_terms:
        return False

    return not bool(previous_terms & rewritten_terms)


def _invalid_rewrite(rewritten_query: str) -> bool:
    cleaned = rewritten_query.strip().lower()
    if cleaned in INVALID_REWRITES:
        return True

    alnum_terms = re.findall(r"[a-z0-9]+", cleaned)
    if not alnum_terms:
        return True

    # Reject tiny punctuation-heavy outputs such as ".*" or "??".
    alnum_chars = sum(len(term) for term in alnum_terms)
    return alnum_chars < 3


def rewrite_query_with_context(
    current_query: str,
    history: list[dict],
    *,
    timeout_seconds: int | None = None,
) -> dict:
    """
    Return {query, used_llm, error}.
    """
    current_query = current_query.strip()
    if not history:
        return {"query": current_query, "used_llm": False, "error": None}

    prompt = f"""
Previous conversation:
{_format_history(history)}

Current user message:
{current_query}

Standalone retrieval query:
""".strip()

    try:
        response = generate_with_fallback(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_output_tokens=120,
            timeout_seconds=timeout_seconds,
        )
        rewritten = " ".join(response.text.split()).strip().strip('"')
        if not rewritten:
            raise LLMProviderError("query rewriter returned an empty query")
        if _invalid_rewrite(rewritten):
            fallback_query = simple_contextual_query(current_query, history)
            return {
                "query": fallback_query,
                "used_llm": False,
                "llm_rewrite_rejected": rewritten,
                "error": "LLM rewrite was invalid",
            }
        if _dropped_history_topic(
            current_query=current_query,
            rewritten_query=rewritten,
            history=history,
        ):
            fallback_query = simple_contextual_query(current_query, history)
            return {
                "query": fallback_query,
                "used_llm": False,
                "llm_rewrite_rejected": rewritten,
                "error": "LLM rewrite dropped the previous topic",
            }
        return {
            "query": rewritten,
            "used_llm": True,
            "provider_used": response.provider,
            "model_used": response.model,
            "fallback_used": response.fallback_used,
            "primary_error": response.primary_error,
            "error": None,
        }
    except Exception as exc:
        return {
            "query": simple_contextual_query(current_query, history),
            "used_llm": False,
            "error": str(exc),
        }
