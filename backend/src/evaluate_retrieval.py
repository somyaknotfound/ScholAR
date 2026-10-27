"""
evaluate_retrieval.py

Small retrieval benchmark for scholAR.

This script checks whether important academic queries retrieve the expected
source chunks near the top of the results. It reports search-style metrics:

    Recall@1 / Recall@3 / Recall@5
        Did a relevant chunk appear within the top K results?

    MRR
        Mean Reciprocal Rank. Higher is better because it rewards the correct
        chunk appearing at rank 1 instead of rank 4 or 5.

Run:
    python backend/src/evaluate_retrieval.py
    python backend/src/evaluate_retrieval.py --modes vector hybrid hybrid_mmr
    python backend/src/evaluate_retrieval.py --show-details
    python backend/src/evaluate_retrieval.py --strict
"""

import argparse
import sys

from search import search_chunks

DEFAULT_TOP_K = 5
DEFAULT_MODES = ["vector", "hybrid", "hybrid_mmr"]
RECALL_CUTOFFS = [1, 3, 5]


# This is intentionally small and hand-written. In retrieval systems, this is
# often called a "gold set": queries where we know what a good result should
# contain. Add new cases whenever you add new PDFs or find a failure case.
EVAL_CASES = [
    {
        "name": "Banker's algorithm",
        "query": "banker's algorithm safe state",
        "expected_source_contains": "Deadlocks",
        "expected_heading_contains": "Banker",
    },
    {
        "name": "Deadlock detection wait-for graph",
        "query": "wait-for graph deadlock detection",
        "expected_source_contains": "Deadlocks",
        "expected_heading_contains": "Deadlock Detection",
        "expected_text_contains": "wait-for graph",
    },
    {
        "name": "Recovery from deadlock",
        "query": "resource preemption recovery from deadlock",
        "expected_source_contains": "Deadlocks",
        "expected_heading_contains": "Recovery from Deadlock",
    },
    {
        "name": "Deadlock prevention",
        "query": "deadlock prevention mutual exclusion hold and wait",
        "expected_source_contains": "Deadlocks",
        "expected_heading_contains": "Deadlock Prevention",
    },
    {
        "name": "FlowDiffusion",
        "query": "flow diffusion dense optical flow",
        "expected_source_contains": "InstantDrag",
        "expected_text_contains": "FlowDiffusion",
    },
    {
        "name": "FlowGen",
        "query": "flow gen architecture optical flow",
        "expected_source_contains": "InstantDrag",
        "expected_text_contains": "FlowGen",
    },
]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _contains_case_insensitive(text: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    return expected.lower() in (text or "").lower()


def _page_overlap(result_metadata: dict, expected_pages: list[int] | None) -> bool:
    if not expected_pages:
        return True

    start_page = int(result_metadata.get("start_page") or 0)
    end_page = int(result_metadata.get("end_page") or start_page)
    result_pages = set(range(start_page, end_page + 1))
    return bool(result_pages & set(expected_pages))


def _matches_expected(result: dict, case: dict) -> bool:
    metadata = result["metadata"]
    document = result.get("document") or ""

    return (
        _contains_case_insensitive(
            metadata.get("source_file"),
            case.get("expected_source_contains"),
        )
        and _contains_case_insensitive(
            metadata.get("section_heading"),
            case.get("expected_heading_contains"),
        )
        and _contains_case_insensitive(
            document,
            case.get("expected_text_contains"),
        )
        and _page_overlap(metadata, case.get("expected_pages"))
    )


def _first_relevant_rank(results: list[dict], case: dict) -> int | None:
    for result in results:
        if _matches_expected(result, case):
            return result["rank"]
    return None


def evaluate_case(case: dict, top_k: int, mode: str) -> dict:
    results = search_chunks(case["query"], top_k=top_k, mode=mode)
    matched_rank = _first_relevant_rank(results, case)

    top_preview = []
    for result in results[: min(3, len(results))]:
        metadata = result["metadata"]
        top_preview.append(
            {
                "rank": result["rank"],
                "source_file": metadata.get("source_file"),
                "pages": f"{metadata.get('start_page')}-{metadata.get('end_page')}",
                "heading": metadata.get("section_heading"),
                "score": result.get("hybrid_score", result.get("vector_score", 0.0)),
            }
        )

    return {
        "case": case,
        "mode": mode,
        "matched_rank": matched_rank,
        "reciprocal_rank": (1 / matched_rank) if matched_rank else 0.0,
        "top_preview": top_preview,
    }


def summarize_mode(case_results: list[dict], cutoffs: list[int]) -> dict:
    total = len(case_results)
    summary = {
        "queries": total,
        "mrr": sum(r["reciprocal_rank"] for r in case_results) / total,
    }

    for cutoff in cutoffs:
        hits = sum(
            1
            for result in case_results
            if result["matched_rank"] is not None and result["matched_rank"] <= cutoff
        )
        summary[f"recall@{cutoff}"] = hits / total

    return summary


def _format_percent(value: float) -> str:
    return f"{value * 100:5.1f}%"


def print_summary(all_results: dict[str, list[dict]], cutoffs: list[int]) -> None:
    print("\nRetrieval Evaluation Summary")
    print("-" * 72)

    header = ["Mode", "Queries", *[f"Recall@{k}" for k in cutoffs], "MRR"]
    print(f"{header[0]:<10} {header[1]:>7} " + " ".join(f"{h:>10}" for h in header[2:]))
    print("-" * 72)

    for mode, results in all_results.items():
        summary = summarize_mode(results, cutoffs)
        recall_values = " ".join(
            f"{_format_percent(summary[f'recall@{k}']):>10}" for k in cutoffs
        )
        print(
            f"{mode:<10} {summary['queries']:>7} "
            f"{recall_values} {summary['mrr']:>10.3f}"
        )


def print_details(all_results: dict[str, list[dict]]) -> None:
    print("\nDetailed Results")
    print("-" * 72)

    for mode, results in all_results.items():
        print(f"\nMode: {mode}")
        for result in results:
            case = result["case"]
            rank = result["matched_rank"]
            status = "PASS" if rank else "FAIL"
            rank_text = f"rank #{rank}" if rank else "not found"
            print(f"{status:<4} | {case['name']} | {rank_text}")
            print(f"       query: {case['query']}")

            if not rank:
                print("       top results:")
                for preview in result["top_preview"]:
                    print(
                        f"         #{preview['rank']} "
                        f"{preview['source_file']} pages {preview['pages']} | "
                        f"{preview['heading']} | score={preview['score']:.3f}"
                    )


def main() -> None:
    _configure_stdout()

    parser = argparse.ArgumentParser(
        description="Evaluate scholAR retrieval quality with Recall@K and MRR."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of search results to inspect per query. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["vector", "hybrid", "hybrid_mmr"],
        default=DEFAULT_MODES,
        help="Retrieval modes to compare. Default: vector hybrid hybrid_mmr.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print per-query pass/fail details and top failed results.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any selected mode misses any query in top-k.",
    )
    args = parser.parse_args()

    cutoffs = [k for k in RECALL_CUTOFFS if k <= args.top_k]
    if args.top_k not in cutoffs:
        cutoffs.append(args.top_k)

    print(
        f"Running {len(EVAL_CASES)} gold query case(s), "
        f"top_k={args.top_k}, modes={', '.join(args.modes)}"
    )

    all_results = {}
    for mode in args.modes:
        all_results[mode] = [
            evaluate_case(case, top_k=args.top_k, mode=mode)
            for case in EVAL_CASES
        ]

    print_summary(all_results, cutoffs)

    if args.show_details:
        print_details(all_results)

    if args.strict:
        missed_any = any(
            result["matched_rank"] is None
            for mode_results in all_results.values()
            for result in mode_results
        )
        if missed_any:
            sys.exit(1)


if __name__ == "__main__":
    main()
