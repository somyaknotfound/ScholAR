"""
generate_questions.py

Generate practice questions from indexed academic material.

This file is a second "study mode" on top of the same retrieval system used by
rag_answer.py:
  - short: exam-style short-answer questions with answers
  - mcq: multiple-choice questions with options, answer, explanation

Run:
    python backend/src/generate_questions.py "deadlock prevention" --type short --count 3
    python backend/src/generate_questions.py "deadlock prevention" --type mcq --count 5
    python backend/src/generate_questions.py "banker's algorithm" --type mcq --difficulty hard
"""

from __future__ import annotations

import argparse
import sys

from llm_provider import LLMProviderError, generate_with_fallback
from output_credibility import compute_output_credibility
from rag_answer import build_context
from search import (
    DEFAULT_MODE,
    assess_retrieval_confidence,
    build_metadata_filters,
    search_chunks,
)


DEFAULT_TOP_K = 5
DEFAULT_COUNT = 3
DEFAULT_DIFFICULTY = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 1800


SYSTEM_PROMPT = """
You are an academic question generator for college students.

Rules:
- Use only the provided context.
- Every answer/explanation must cite the source using bracket citations like [1].
- Do not invent facts outside the context.
- If the context is insufficient, say so instead of guessing.
- Keep questions clear, exam-relevant, and useful for revision.
""".strip()


QUESTION_PROMPTS = {
    "short": """
Generate {count} {difficulty}-difficulty short-answer practice questions on the topic.

For each question, use this exact structure:

Q1. <question>
Answer: <concise answer with citation>

Requirements:
- Questions should test understanding, not just copy a sentence.
- Answers should be 2-4 sentences unless the concept is very small.
- Include citations in every answer.
""".strip(),
    "mcq": """
Generate {count} {difficulty}-difficulty multiple-choice questions on the topic.

For each question, use this exact structure:

Q1. <question>
A. <option>
B. <option>
C. <option>
D. <option>
Correct answer: <letter>
Explanation: <why the correct option is right, with citation>

Requirements:
- Exactly 4 options per question.
- Exactly one correct answer.
- Distractors should be plausible but clearly incorrect.
- Include citations in every explanation.
""".strip(),
}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def build_question_prompt(
    *,
    topic: str,
    context: str,
    question_type: str,
    count: int,
    difficulty: str,
) -> str:
    task_prompt = QUESTION_PROMPTS[question_type].format(
        count=count,
        difficulty=difficulty,
    )

    return f"""
Topic:
{topic}

Retrieved context:
{context}

Task:
{task_prompt}
""".strip()


def generate_questions(
    topic: str,
    *,
    question_type: str = "short",
    count: int = DEFAULT_COUNT,
    difficulty: str = DEFAULT_DIFFICULTY,
    top_k: int = DEFAULT_TOP_K,
    source_file: str | None = None,
    metadata_filters: dict | None = None,
    mode: str = DEFAULT_MODE,
    temperature: float = 0.35,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Generate study questions from retrieved academic context.
    """
    if question_type not in QUESTION_PROMPTS:
        raise ValueError("question_type must be 'short' or 'mcq'.")
    if count < 1:
        raise ValueError("count must be at least 1.")

    _log("Retrieving chunks...", verbose)
    results = search_chunks(
        query_text=topic,
        top_k=top_k,
        source_file=source_file,
        metadata_filters=metadata_filters,
        mode=mode,
    )
    _log(f"Retrieved {len(results)} chunk(s). Checking confidence...", verbose)

    confidence = assess_retrieval_confidence(results, mode=mode)
    context, sources = build_context(results)

    if not confidence["confident"]:
        questions = (
            "I could not find enough reliable information in the indexed "
            "materials to generate useful practice questions. Try a more "
            "specific topic or add more relevant documents."
        )
        _log("Retrieval is weak. Skipping question generation.", verbose)
        return {
            "topic": topic,
            "question_type": question_type,
            "questions": questions,
            "retrieval_confidence": confidence,
            "output_credibility": compute_output_credibility(
                generated_text=questions,
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
            user_prompt=build_question_prompt(
                topic=topic,
                context=context,
                question_type=question_type,
                count=count,
                difficulty=difficulty,
            ),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        _log(f"Questions generated by {llm_response.provider}.", verbose)
    except LLMProviderError as exc:
        questions = (
            "Retrieval succeeded, but question generation failed because no "
            "configured LLM provider completed the request."
        )
        return {
            "topic": topic,
            "question_type": question_type,
            "questions": questions,
            "retrieval_confidence": confidence,
            "output_credibility": compute_output_credibility(
                generated_text=questions,
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
        "topic": topic,
        "question_type": question_type,
        "questions": llm_response.text,
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


def print_question_response(
    response: dict,
    *,
    show_sources: bool = True,
    show_context: bool = False,
) -> None:
    confidence = response["retrieval_confidence"]
    confidence_label = "confident" if confidence["confident"] else "weak"

    print(f'Topic: "{response["topic"]}"')
    print(f"Question type: {response['question_type']}")
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
            f"Question set credibility: {credibility['label']} "
            f"({credibility['score']}/100)"
        )

    print("\nGenerated questions:")
    print(response["questions"])

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
        description="Generate practice questions from the local academic resource index."
    )
    parser.add_argument(
        "topic",
        nargs="*",
        help="Topic to generate questions from.",
    )
    parser.add_argument(
        "--type",
        choices=["short", "mcq"],
        default="short",
        help="Question type to generate. Default: short.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of questions to generate. Default: {DEFAULT_COUNT}.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=DEFAULT_DIFFICULTY,
        help=f"Question difficulty. Default: {DEFAULT_DIFFICULTY}.",
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
        help="Optional exact source_file filter, e.g. Deadlocks.pdf.",
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
        default=0.35,
        help="LLM temperature. Default: 0.35.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum generation length requested from the LLM. "
            f"Default: {DEFAULT_MAX_OUTPUT_TOKENS}."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Maximum seconds to wait per LLM provider before fallback.",
    )
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Do not print the source list after generated questions.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the exact retrieved context sent to the LLM.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while retrieval and LLM calls run.",
    )
    args = parser.parse_args()

    topic = " ".join(args.topic).strip()
    if not topic:
        topic = input("Enter a topic: ").strip()

    metadata_filters = build_metadata_filters(
        subject=args.subject,
        doc_type=args.doc_type,
        source_type=args.source_type,
        min_credibility=args.min_credibility,
    )

    response = generate_questions(
        topic,
        question_type=args.type,
        count=args.count,
        difficulty=args.difficulty,
        top_k=args.top_k,
        source_file=args.source,
        metadata_filters=metadata_filters,
        mode=args.mode,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        verbose=args.verbose,
    )
    print_question_response(
        response,
        show_sources=not args.hide_sources,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    main()
