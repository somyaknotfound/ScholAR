"""
output_credibility.py

Compute a simple, explainable credibility score for generated outputs.

This is different from document/source credibility:
  - source credibility: how trustworthy a PDF/chunk is
  - output credibility: how trustworthy this generated answer/question set is

Version 1 intentionally uses a small set of transparent signals:
  40% retrieval strength       -> did search retrieve strong evidence?
  30% average source credibility -> are the sources trustworthy?
  15% source variety           -> did the output cite multiple sources?
  15% citation usage           -> did it frequently ground claims with citations?
"""

from __future__ import annotations

import re


def extract_citation_numbers(text: str) -> list[int]:
    """
    Extract bracket citations like [1], [2], [3] from generated text.
    """
    return [int(match) for match in re.findall(r"\[(\d+)\]", text or "")]


def _label(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def _retrieval_strength(retrieval_confidence: dict) -> float:
    """
    Use the top retrieval score already computed by search.py.

    In hybrid mode this is the top hybrid score. This slightly includes source
    credibility because hybrid ranking has a 10% credibility component; we
    accept that small overlap in v1 for simplicity.
    """
    if not retrieval_confidence.get("confident"):
        return 0.0
    score = retrieval_confidence.get("top_score", 0.0) or 0.0
    return max(0.0, min(1.0, float(score)))


def _average_source_credibility(sources: list[dict]) -> float:
    scores = [
        float(source.get("credibility_score") or 0)
        for source in sources
    ]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, (sum(scores) / len(scores)) / 100))


def _valid_citations(answer_text: str, source_count: int) -> list[int]:
    citations = extract_citation_numbers(answer_text)
    return [
        citation
        for citation in citations
        if 1 <= citation <= source_count
    ]


def _source_variety(valid_citations: list[int]) -> float:
    """
    Reward using multiple different cited sources.
    """
    unique_count = len(set(valid_citations))
    if unique_count == 0:
        return 0.0
    if unique_count == 1:
        return 0.60
    if unique_count == 2:
        return 0.85
    return 1.0


def _citation_usage(valid_citations: list[int]) -> float:
    """
    Reward grounding multiple claims with citations.
    """
    count = len(valid_citations)
    if count == 0:
        return 0.0
    if count == 1:
        return 0.50
    if count == 2:
        return 0.75
    return 1.0


def compute_output_credibility(
    *,
    generated_text: str,
    retrieval_confidence: dict,
    sources: list[dict],
) -> dict:
    """
    Return an explainable 0-100 output credibility score.
    """
    valid_citations = _valid_citations(generated_text, len(sources))
    unique_valid_citations = sorted(set(valid_citations))

    retrieval = _retrieval_strength(retrieval_confidence)
    source_credibility = _average_source_credibility(sources)
    variety = _source_variety(valid_citations)
    citation_usage = _citation_usage(valid_citations)

    score_float = (
        (0.40 * retrieval)
        + (0.30 * source_credibility)
        + (0.15 * variety)
        + (0.15 * citation_usage)
    )
    score = round(score_float * 100)

    reasons = [
        f"retrieval strength: {retrieval:.2f}",
        f"average source credibility: {round(source_credibility * 100)}/100",
        f"unique cited sources: {len(unique_valid_citations)}",
        f"valid citation uses: {len(valid_citations)}",
    ]

    if not retrieval_confidence.get("confident"):
        reasons.insert(0, "retrieval was weak, so generation was not trusted")

    return {
        "score": score,
        "label": _label(score),
        "signals": {
            "retrieval_strength": retrieval,
            "average_source_credibility": source_credibility,
            "source_variety": variety,
            "citation_usage": citation_usage,
            "valid_citation_count": len(valid_citations),
            "unique_cited_source_count": len(unique_valid_citations),
        },
        "reasons": reasons,
    }
