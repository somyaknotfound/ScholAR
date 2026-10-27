"""
evaluate_rag.py

Small RAG sanity check.

This is not an LLM-as-judge benchmark. It is a practical regression test that
checks whether the RAG layer behaves safely:
  - good academic queries retrieve confident context
  - generated answers contain citations
  - returned citations point to available sources
  - weak/off-topic queries refuse instead of calling the LLM

Run:
    python backend/src/evaluate_rag.py
    python backend/src/evaluate_rag.py --show-answers
    python backend/src/evaluate_rag.py --timeout-seconds 30
"""

from __future__ import annotations

import argparse
import re
import sys

from rag_answer import answer_question


DEFAULT_TOP_K = 3
DEFAULT_TIMEOUT_SECONDS = 45


EVAL_CASES = [
    {
        "name": "deadlock prevention cited answer",
        "query": "explain deadlock prevention",
        "expect_confident": True,
        "expect_citations": True,
        "expected_source_contains": "Deadlocks.pdf",
        "expected_answer_terms": ["deadlock", "prevention"],
    },
    {
        "name": "flow diffusion cited answer",
        "query": "flow diffusion architecture",
        "expect_confident": True,
        "expect_citations": True,
        "expected_source_contains": "InstantDrag.pdf",
        "expected_answer_terms": ["flow", "diffusion"],
    },
    {
        "name": "weak query refusal",
        "query": "pizza recipe chocolate cake",
        "expect_confident": False,
        "expect_citations": False,
        "expected_provider": None,
        "expected_answer_terms": ["could not find enough"],
    },
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _citations(answer: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", answer)]


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _source_matches(response: dict, expected: str | None) -> bool:
    if not expected:
        return True
    expected_lower = expected.lower()
    return any(
        expected_lower in (source.get("source_file") or "").lower()
        for source in response.get("sources", [])
    )


def _citations_are_valid(answer: str, sources: list[dict]) -> bool:
    citation_numbers = _citations(answer)
    if not citation_numbers:
        return False
    source_count = len(sources)
    return all(1 <= number <= source_count for number in citation_numbers)


def evaluate_case(
    case: dict,
    *,
    top_k: int,
    mode: str,
    timeout_seconds: int,
    verbose: bool,
) -> dict:
    response = answer_question(
        case["query"],
        top_k=top_k,
        mode=mode,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
    )

    confidence = response["retrieval_confidence"]
    answer = response["answer"]
    sources = response.get("sources", [])

    checks = []

    expected_confident = case["expect_confident"]
    checks.append(
        {
            "name": "retrieval confidence",
            "passed": confidence["confident"] == expected_confident,
            "detail": (
                f"expected {expected_confident}, "
                f"got {confidence['confident']} ({confidence['reason']})"
            ),
        }
    )

    if case.get("expect_citations"):
        checks.append(
            {
                "name": "answer has valid citations",
                "passed": _citations_are_valid(answer, sources),
                "detail": f"citations={_citations(answer)}, source_count={len(sources)}",
            }
        )
    else:
        checks.append(
            {
                "name": "weak answer avoids citations",
                "passed": not _citations(answer),
                "detail": f"citations={_citations(answer)}",
            }
        )

    if expected_confident:
        checks.append(
            {
                "name": "LLM provider used",
                "passed": bool(response.get("provider_used")) and not response.get("llm_error"),
                "detail": (
                    f"provider={response.get('provider_used')}, "
                    f"fallback={response.get('fallback_used')}, "
                    f"llm_error={response.get('llm_error')}"
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "LLM skipped for weak retrieval",
                "passed": response.get("provider_used") is None,
                "detail": f"provider={response.get('provider_used')}",
            }
        )

    checks.append(
        {
            "name": "expected source present",
            "passed": _source_matches(response, case.get("expected_source_contains")),
            "detail": f"expected source contains {case.get('expected_source_contains')}",
        }
    )

    checks.append(
        {
            "name": "answer contains expected terms",
            "passed": _contains_all_terms(answer, case.get("expected_answer_terms", [])),
            "detail": f"terms={case.get('expected_answer_terms', [])}",
        }
    )

    output_credibility = response.get("output_credibility") or {}
    checks.append(
        {
            "name": "output credibility present",
            "passed": (
                isinstance(output_credibility.get("score"), int)
                and output_credibility.get("label") in {"low", "medium", "high"}
            ),
            "detail": (
                f"label={output_credibility.get('label')}, "
                f"score={output_credibility.get('score')}"
            ),
        }
    )

    passed = all(check["passed"] for check in checks)

    return {
        "passed": passed,
        "case": case,
        "checks": checks,
        "response": response,
    }


def _print_result(result: dict, *, show_answers: bool) -> None:
    case = result["case"]
    response = result["response"]
    status = "PASS" if result["passed"] else "FAIL"

    print(f"{status} | {case['name']}")
    print(f"     query: {case['query']}")
    print(
        f"     confidence: {response['retrieval_confidence']['confident']} "
        f"({response['retrieval_confidence']['reason']})"
    )
    print(
        f"     provider: {response.get('provider_used')} "
        f"| fallback: {response.get('fallback_used')}"
    )
    if response.get("output_credibility"):
        credibility = response["output_credibility"]
        print(
            f"     output credibility: {credibility['label']} "
            f"({credibility['score']}/100)"
        )

    for check in result["checks"]:
        check_status = "ok" if check["passed"] else "bad"
        print(f"     [{check_status}] {check['name']}: {check['detail']}")

    if show_answers:
        print("     answer:")
        for line in response["answer"].splitlines():
            print(f"       {line}")


def main() -> None:
    _configure_stdout()

    parser = argparse.ArgumentParser(
        description="Run basic RAG behavior checks."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per case. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode to evaluate. Default: hybrid.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Maximum seconds to wait per LLM provider. "
            f"Default: {DEFAULT_TIMEOUT_SECONDS}."
        ),
    )
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="Print full generated answers for inspection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress from the underlying RAG pipeline.",
    )
    args = parser.parse_args()

    print(
        f"Running {len(EVAL_CASES)} RAG check(s), "
        f"top_k={args.top_k}, mode={args.mode}, "
        f"timeout={args.timeout_seconds}s"
    )
    print("-" * 70)

    passed_count = 0
    for case in EVAL_CASES:
        result = evaluate_case(
            case,
            top_k=args.top_k,
            mode=args.mode,
            timeout_seconds=args.timeout_seconds,
            verbose=args.verbose,
        )
        if result["passed"]:
            passed_count += 1
        _print_result(result, show_answers=args.show_answers)
        print()

    total = len(EVAL_CASES)
    print("-" * 70)
    print(f"Passed {passed_count}/{total} RAG check(s).")

    if passed_count != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
