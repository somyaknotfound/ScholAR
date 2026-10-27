"""
rag_answer.py

Answer questions using the local retrieval index plus an LLM.

Pipeline:
  1. Hybrid search retrieves source chunks from Chroma/BM25.
  2. Retrieval confidence gate blocks weak context.
  3. A citation-aware prompt is built from the retrieved chunks.
  4. llm_provider.py tries the primary LLM, then fallback provider if needed.

Run:
  python backend/src/rag_answer.py "explain deadlock prevention"
  python backend/src/rag_answer.py "what is FlowGen?" --top-k 3
  python backend/src/rag_answer.py "banker's algorithm" --source DeadLocks.pdf
"""

from __future__ import annotations

import argparse
import sys

from llm_provider import LLMProviderError, generate_with_fallback
from output_credibility import compute_output_credibility
from search import (
    DEFAULT_MODE,
    assess_retrieval_confidence,
    build_metadata_filters,
    search_chunks,
)


DEFAULT_TOP_K = 5
MAX_CHARS_PER_CHUNK = 1400
DEFAULT_MAX_OUTPUT_TOKENS = 1400


SYSTEM_PROMPT = """
You are an academic study assistant.

Rules:
- Answer only using the provided context.
- Cite every factual claim using bracket citations like [1] or [2].
- If the context is insufficient, say what is missing instead of guessing.
- Prefer clear student-friendly explanations.
- Keep the answer complete; do not stop mid-sentence.
- Do not invent sources, page numbers, or citations.
""".strip()


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _clean_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _source_label(metadata: dict) -> str:
    source_file = metadata.get("source_file", "unknown source")
    start_page = metadata.get("start_page", "unknown")
    end_page = metadata.get("end_page", start_page)
    heading = metadata.get("section_heading", "unknown")
    credibility_label = metadata.get("credibility_label", "unknown")
    credibility_score = metadata.get("credibility_score", 0)
    return (
        f"{source_file}, pages {start_page}-{end_page}, "
        f"heading: {heading}, credibility: {credibility_label} "
        f"({credibility_score}/100)"
    )


def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """
    Convert search results into numbered context blocks and source metadata.
    """
    context_blocks = []
    sources = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        label = _source_label(metadata)
        text = _clean_text(result["document"])

        context_blocks.append(
            f"[{index}] {label}\n"
            f"Content: {text}"
        )

        sources.append(
            {
                "citation": f"[{index}]",
                "chunk_id": result["id"],
                "source_file": metadata.get("source_file"),
                "start_page": metadata.get("start_page"),
                "end_page": metadata.get("end_page"),
                "section_heading": metadata.get("section_heading"),
                "document_id": metadata.get("document_id"),
                "source_path": metadata.get("source_path"),
                "credibility_score": metadata.get("credibility_score"),
                "credibility_label": metadata.get("credibility_label"),
                "retrieval_score": result.get("hybrid_score", result.get("vector_score")),
            }
        )

    return "\n\n".join(context_blocks), sources


def _format_conversation_context(conversation_context: list[dict] | None) -> str:
    if not conversation_context:
        return "No previous conversation."

    lines = []
    for message in conversation_context:
        role = message.get("role", "unknown")
        content = " ".join((message.get("content") or "").split())
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


def build_user_prompt(
    query: str,
    context: str,
    *,
    conversation_context: list[dict] | None = None,
    retrieval_query: str | None = None,
) -> str:
    retrieval_note = ""
    if retrieval_query and retrieval_query.strip() != query.strip():
        retrieval_note = f"\nStandalone retrieval query used:\n{retrieval_query}\n"

    return f"""
Previous conversation:
{_format_conversation_context(conversation_context)}
{retrieval_note}
Question:
{query}

Retrieved context:
{context}

Answer with citations:
""".strip()


def answer_question(
    query: str,
    *,
    retrieval_query: str | None = None,
    conversation_context: list[dict] | None = None,
    top_k: int = DEFAULT_TOP_K,
    source_file: str | None = None,
    metadata_filters: dict | None = None,
    mode: str = DEFAULT_MODE,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    End-to-end RAG answer function for CLI now and API/UI later.
    """
    search_query = retrieval_query or query
    _log("Retrieving chunks...", verbose)
    results = search_chunks(
        query_text=search_query,
        top_k=top_k,
        source_file=source_file,
        metadata_filters=metadata_filters,
        mode=mode,
    )
    _log(f"Retrieved {len(results)} chunk(s). Checking confidence...", verbose)
    confidence = assess_retrieval_confidence(results, mode=mode)
    context, sources = build_context(results)

    if not confidence["confident"]:
        answer = (
            "I could not find enough reliable information in the indexed "
            "materials to answer this confidently. Try a more specific "
            "query or add more relevant documents."
        )
        _log("Retrieval is weak. Skipping LLM call.", verbose)
        return {
            "query": query,
            "retrieval_query": search_query,
            "answer": answer,
            "retrieval_confidence": confidence,
            "output_credibility": compute_output_credibility(
                generated_text=answer,
                retrieval_confidence=confidence,
                sources=sources,
            ),
            "context": context,
            "sources": sources,
            "provider_used": None,
            "model_used": None,
            "fallback_used": False,
            "llm_error": None,
        }

    try:
        _log("Calling LLM provider(s)...", verbose)
        llm_response = generate_with_fallback(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                query,
                context,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
            ),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        _log(
            f"LLM response received from {llm_response.provider}.",
            verbose,
        )
    except LLMProviderError as exc:
        answer = (
            "Retrieval succeeded, but answer generation failed because no "
            "configured LLM provider completed the request."
        )
        return {
            "query": query,
            "retrieval_query": search_query,
            "answer": answer,
            "retrieval_confidence": confidence,
            "output_credibility": compute_output_credibility(
                generated_text=answer,
                retrieval_confidence=confidence,
                sources=sources,
            ),
            "context": context,
            "sources": sources,
            "provider_used": None,
            "model_used": None,
            "fallback_used": False,
            "llm_error": str(exc),
        }

    return {
        "query": query,
        "retrieval_query": search_query,
        "answer": llm_response.text,
        "retrieval_confidence": confidence,
        "output_credibility": compute_output_credibility(
            generated_text=llm_response.text,
            retrieval_confidence=confidence,
            sources=sources,
        ),
        "context": context,
        "sources": sources,
        "provider_used": llm_response.provider,
        "model_used": llm_response.model,
        "fallback_used": llm_response.fallback_used,
        "primary_error": llm_response.primary_error,
        "llm_error": None,
    }


def print_rag_response(
    response: dict,
    *,
    show_sources: bool = True,
    show_context: bool = False,
) -> None:
    confidence = response["retrieval_confidence"]
    confidence_label = "confident" if confidence["confident"] else "weak"

    print(f'Query: "{response["query"]}"')
    print(f"Retrieval confidence: {confidence_label} ({confidence['reason']})")

    if response.get("provider_used"):
        print(
            f"LLM: {response['provider_used']} / {response['model_used']} "
            f"| fallback_used={response['fallback_used']}"
        )
    if response.get("primary_error"):
        print(f"Primary provider error: {response['primary_error']}")
    if response.get("llm_error"):
        print(f"LLM error: {response['llm_error']}")
    if response.get("output_credibility"):
        credibility = response["output_credibility"]
        print(
            f"Answer credibility: {credibility['label']} "
            f"({credibility['score']}/100)"
        )

    print("\nAnswer:")
    print(response["answer"])

    if show_context and response.get("context"):
        print("\nRetrieved context sent to LLM:")
        print(response["context"])

    if show_sources and response["sources"]:
        print("\nSources:")
        for source in response["sources"]:
            print(
                f"  {source['citation']} {source['source_file']} "
                f"pages {source['start_page']}-{source['end_page']} "
                f"| heading: {source['section_heading']} "
                f"| credibility: {source['credibility_label']} "
                f"({source['credibility_score']}/100)"
            )


def main() -> None:
    _configure_stdout()

    parser = argparse.ArgumentParser(
        description="Generate a cited RAG answer from the local academic resource index."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to answer using indexed material.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--source",
        help="Optional exact source_file filter, e.g. DeadLocks.pdf.",
    )
    parser.add_argument(
        "--subject",
        help='Optional exact subject filter, e.g. "Operating Systems".',
    )
    parser.add_argument(
        "--doc-type",
        help="Optional exact document type filter, e.g. lecture_notes.",
    )
    parser.add_argument(
        "--source-type",
        help="Optional exact source type filter, e.g. faculty_notes.",
    )
    parser.add_argument(
        "--min-credibility",
        type=int,
        help="Optional minimum source credibility score, e.g. 70.",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "hybrid_mmr"],
        default=DEFAULT_MODE,
        help=f"Retrieval mode. Default: {DEFAULT_MODE}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature. Lower is more factual. Default: 0.2.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum answer length requested from the LLM. "
            f"Default: {DEFAULT_MAX_OUTPUT_TOKENS}."
        ),
    )
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Do not print the source list after the answer.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the exact retrieved context sent to the LLM.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Maximum seconds to wait per LLM provider before fallback.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while retrieval and LLM calls run.",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Enter a question: ").strip()

    metadata_filters = build_metadata_filters(
        subject=args.subject,
        doc_type=args.doc_type,
        source_type=args.source_type,
        min_credibility=args.min_credibility,
    )

    response = answer_question(
        query,
        top_k=args.top_k,
        source_file=args.source,
        metadata_filters=metadata_filters,
        mode=args.mode,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        verbose=args.verbose,
    )
    print_rag_response(
        response,
        show_sources=not args.hide_sources,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    main()
