"""
search.py

Query the local ChromaDB index and print sourced search results.

Retrieval modes:
    vector: dense semantic similarity only
    hybrid: vector similarity + BM25 + keyword overlap + heading match
    hybrid_mmr: hybrid candidates re-ranked for diversity with MMR

Run:
    python search.py "deadlock prevention"
    python search.py "banker's algorithm safe state" --top-k 3
    python search.py "deadlock detection" --source DeadLocks.pdf
    python search.py "wait-for graph" --mode vector
"""

import argparse
import re
import sys

from bm25_index import load_or_build_bm25_index, records_by_id, score_bm25, tokenize
from embed_index import get_chroma_collection, get_model

DEFAULT_TOP_K = 5
DEFAULT_MODE = "hybrid"
PREVIEW_CHARS = 300
HYBRID_CANDIDATE_MULTIPLIER = 5
MIN_HYBRID_CANDIDATES = 20
MMR_LAMBDA = 0.75
MIN_CONFIDENT_VECTOR_SCORE = 0.45
MIN_CONFIDENT_HYBRID_SCORE = 0.40
MIN_CONFIDENT_KEYWORD_SCORE = 0.20


def _configure_stdout() -> None:
    """
    Keep Unicode output readable on Windows terminals when possible.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    """
    Collapse whitespace so terminal output is readable, then truncate.
    """
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def build_metadata_filters(
    *,
    subject: str | None = None,
    doc_type: str | None = None,
    source_type: str | None = None,
    min_credibility: int | None = None,
) -> dict:
    """
    Build metadata filters used by both Chroma vector search and BM25 search.
    """
    filters = {}
    if subject:
        filters["subject"] = subject
    if doc_type:
        filters["doc_type"] = doc_type
    if source_type:
        filters["source_type"] = source_type
    if min_credibility is not None:
        filters["min_credibility"] = min_credibility
    return filters


def _chroma_where(source_file: str | None, metadata_filters: dict) -> dict | None:
    """
    Convert our filter dict into a Chroma where clause.
    """
    conditions = []
    if source_file:
        conditions.append({"source_file": source_file})
    if metadata_filters.get("subject"):
        conditions.append({"subject": metadata_filters["subject"]})
    if metadata_filters.get("doc_type"):
        conditions.append({"doc_type": metadata_filters["doc_type"]})
    if metadata_filters.get("source_type"):
        conditions.append({"source_type": metadata_filters["source_type"]})
    if metadata_filters.get("min_credibility") is not None:
        conditions.append({
            "credibility_score": {"$gte": metadata_filters["min_credibility"]}
        })

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _compact_alnum(text: str) -> str:
    """
    Remove punctuation/spacing for compact term checks.

    This helps a query like "flow gen" match "FlowGen" without making every
    single partial word overlap look important.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _vector_score(distance: float) -> float:
    """
    Convert cosine distance into a rough 0..1 similarity score.

    Chroma returns lower distance for better matches. With cosine distance,
    1 - distance is a useful simple similarity-like value for ranking display
    and hybrid scoring.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def _keyword_score(query_text: str, document: str, heading: str | None) -> float:
    """
    Score exact keyword overlap between query and a candidate chunk.

    Heading matches are counted here too because academic queries often name a
    topic exactly: "Banker's Algorithm", "wait-for graph", "circular wait".
    """
    query_tokens = tokenize(query_text)
    query_terms = set(query_tokens)
    if not query_terms:
        return 0.0

    document_terms = set(tokenize(document))
    heading_terms = set(tokenize(heading or ""))

    text_hits = len(query_terms & document_terms)
    heading_hits = len(query_terms & heading_terms)

    text_coverage = text_hits / len(query_terms)
    heading_coverage = heading_hits / len(query_terms)

    # Damp partial overlap: one matching word in a multi-word query should help,
    # but it should not receive a large lexical boost by itself.
    overlap_score = ((0.75 * text_coverage) + (0.25 * heading_coverage)) ** 2

    normalized_query = " ".join(tokenize(query_text))
    normalized_document = " ".join(tokenize(document))
    normalized_heading = " ".join(tokenize(heading or ""))

    exact_phrase_bonus = 0.0
    if normalized_query and normalized_query in normalized_document:
        exact_phrase_bonus += 0.20
    if normalized_query and normalized_query in normalized_heading:
        exact_phrase_bonus += 0.15

    compact_query = "".join(query_tokens)
    if len(query_tokens) >= 2 and compact_query:
        if compact_query in _compact_alnum(document):
            exact_phrase_bonus += 0.25
        if compact_query in _compact_alnum(heading or ""):
            exact_phrase_bonus += 0.15

    return min(1.0, overlap_score + exact_phrase_bonus)


def _heading_score(query_text: str, heading: str | None) -> float:
    """
    Separate heading boost for topic-title matches.
    """
    query_terms = set(tokenize(query_text))
    heading_terms = set(tokenize(heading or ""))
    if not query_terms or not heading_terms:
        return 0.0
    return len(query_terms & heading_terms) / len(query_terms)


def _hybrid_score(
    query_text: str,
    document: str,
    metadata: dict,
    distance: float,
    bm25_score: float = 0.0,
) -> dict:
    """
    Combine dense similarity with lightweight lexical signals.

    Weights are deliberately simple and explainable:
        60% vector similarity
        15% BM25 lexical relevance
        5% keyword overlap
        10% heading match
        10% credibility
    """
    heading = metadata.get("section_heading")
    vector = _vector_score(distance)
    keyword = _keyword_score(query_text, document, heading)
    heading_match = _heading_score(query_text, heading)
    credibility = (metadata.get("credibility_score") or 0) / 100
    final = (
        (0.60 * vector)
        + (0.15 * bm25_score)
        + (0.05 * keyword)
        + (0.10 * heading_match)
        + (0.10 * credibility)
    )

    return {
        "vector_score": vector,
        "bm25_score": bm25_score,
        "keyword_score": keyword,
        "heading_score": heading_match,
        "credibility_score_normalized": credibility,
        "hybrid_score": final,
    }


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _apply_mmr(
    candidates: list[dict],
    top_k: int,
    *,
    lambda_mult: float = MMR_LAMBDA,
) -> list[dict]:
    """
    Select a relevant but less-redundant subset of already-ranked candidates.

    MMR = lambda * relevance - (1 - lambda) * similarity_to_selected

    In this project, relevance is the existing hybrid score. Similarity is
    cosine similarity between candidate chunk embeddings. Since embeddings are
    normalized, cosine similarity is just a dot product.
    """
    if len(candidates) <= top_k:
        for candidate in candidates:
            candidate["mmr_score"] = candidate.get("hybrid_score", 0.0)
            candidate["mmr_diversity_penalty"] = 0.0
        return candidates

    model = get_model()
    texts = [candidate["document"] for candidate in candidates]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    selected_indexes = []
    remaining_indexes = set(range(len(candidates)))

    # Anchor the selection with the most relevant chunk. This keeps MMR from
    # getting too adventurous on academic queries where the top hit matters.
    first_index = max(
        remaining_indexes,
        key=lambda index: candidates[index].get("hybrid_score", 0.0),
    )
    selected_indexes.append(first_index)
    remaining_indexes.remove(first_index)
    candidates[first_index]["mmr_score"] = candidates[first_index].get("hybrid_score", 0.0)
    candidates[first_index]["mmr_diversity_penalty"] = 0.0

    while remaining_indexes and len(selected_indexes) < top_k:
        best_index = None
        best_score = float("-inf")
        best_penalty = 0.0

        for index in remaining_indexes:
            relevance = candidates[index].get("hybrid_score", 0.0)
            similarity_to_selected = max(
                _dot_product(embeddings[index], embeddings[selected_index])
                for selected_index in selected_indexes
            )
            mmr_score = (
                (lambda_mult * relevance)
                - ((1 - lambda_mult) * similarity_to_selected)
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
                best_penalty = similarity_to_selected

        selected_indexes.append(best_index)
        remaining_indexes.remove(best_index)
        candidates[best_index]["mmr_score"] = best_score
        candidates[best_index]["mmr_diversity_penalty"] = best_penalty

    return [candidates[index] for index in selected_indexes]


def search_chunks(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    source_file: str | None = None,
    mode: str = DEFAULT_MODE,
    metadata_filters: dict | None = None,
) -> list[dict]:
    """
    Return the top search matches for a query.

    Each returned dict contains:
        rank, id, distance, document, metadata, and score fields
    """
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("Query text cannot be empty.")
    if mode not in {"vector", "hybrid", "hybrid_mmr"}:
        raise ValueError("mode must be 'vector', 'hybrid', or 'hybrid_mmr'.")
    metadata_filters = metadata_filters or {}

    collection = get_chroma_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    model = get_model()
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    candidate_count = top_k
    if mode in {"hybrid", "hybrid_mmr"}:
        candidate_count = max(top_k * HYBRID_CANDIDATE_MULTIPLIER, MIN_HYBRID_CANDIDATES)

    query_args = {
        "query_embeddings": query_embedding,
        "n_results": min(candidate_count, total_chunks),
        "include": ["documents", "metadatas", "distances"],
    }
    where = _chroma_where(source_file, metadata_filters)
    if where:
        query_args["where"] = where

    raw_results = collection.query(**query_args)

    ids = raw_results.get("ids", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results_by_id = {}
    for index, chunk_id in enumerate(ids):
        results_by_id[chunk_id] = {
            "rank": index + 1,
            "id": chunk_id,
            "distance": distances[index],
            "document": documents[index],
            "metadata": metadatas[index],
        }

    bm25_scores = {}
    if mode in {"hybrid", "hybrid_mmr"}:
        bm25_index = load_or_build_bm25_index(collection)
        bm25_scores = score_bm25(
            query_text,
            bm25_index,
            source_file=source_file,
            metadata_filters=metadata_filters,
        )
        bm25_candidate_ids = [
            chunk_id
            for chunk_id, score in sorted(
                bm25_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ][:candidate_count]

        bm25_records_by_id = records_by_id(bm25_index)
        for chunk_id in bm25_candidate_ids:
            if chunk_id in results_by_id:
                continue
            record = bm25_records_by_id[chunk_id]
            # BM25-only candidates do not have a vector distance from the
            # Chroma query result. Use distance=1.0 so vector_score=0.0, while
            # strong lexical matches can still contribute through BM25/keyword.
            results_by_id[chunk_id] = {
                "rank": 0,
                "id": chunk_id,
                "distance": 1.0,
                "document": record["document"],
                "metadata": record["metadata"],
            }

    results = []
    for chunk_id, result in results_by_id.items():
        scores = _hybrid_score(
            query_text=query_text,
            document=result["document"],
            metadata=result["metadata"],
            distance=result["distance"],
            bm25_score=bm25_scores.get(chunk_id, 0.0),
        )
        result.update(scores)
        results.append(result)

    if mode in {"hybrid", "hybrid_mmr"}:
        results.sort(key=lambda result: result["hybrid_score"], reverse=True)
        if mode == "hybrid_mmr":
            results = _apply_mmr(results, top_k)
        results = results[:top_k]

    for index, result in enumerate(results):
        result["rank"] = index + 1

    return results


def assess_retrieval_confidence(results: list[dict], mode: str = DEFAULT_MODE) -> dict:
    """
    Decide whether retrieved chunks are strong enough to trust downstream.

    This is mainly for RAG: if retrieval is weak, the answer layer should
    refuse/fallback instead of asking an LLM to answer from poor context.
    """
    if not results:
        return {
            "confident": False,
            "reason": "no results returned",
            "top_score": 0.0,
        }

    top = results[0]
    vector = top.get("vector_score", 0.0)
    bm25 = top.get("bm25_score", 0.0)
    keyword = top.get("keyword_score", 0.0)
    hybrid = top.get("hybrid_score", vector)

    if mode == "vector":
        confident = vector >= MIN_CONFIDENT_VECTOR_SCORE
        reason = (
            "top vector similarity is strong enough"
            if confident
            else "top vector similarity is below threshold"
        )
        return {
            "confident": confident,
            "reason": reason,
            "top_score": vector,
            "top_vector_score": vector,
            "top_bm25_score": bm25,
            "top_keyword_score": keyword,
            "top_hybrid_score": hybrid,
        }

    confident = (
        hybrid >= MIN_CONFIDENT_HYBRID_SCORE
        and (
            vector >= MIN_CONFIDENT_VECTOR_SCORE
            or bm25 >= MIN_CONFIDENT_KEYWORD_SCORE
            or keyword >= MIN_CONFIDENT_KEYWORD_SCORE
        )
    )
    if confident:
        reason = "top hybrid result has sufficient semantic or keyword support"
    else:
        reason = "top hybrid result is below confidence thresholds"

    return {
        "confident": confident,
        "reason": reason,
        "top_score": hybrid,
        "top_vector_score": vector,
        "top_bm25_score": bm25,
        "top_keyword_score": keyword,
        "top_hybrid_score": hybrid,
    }


def print_results(query_text: str, results: list[dict], top_k: int, mode: str) -> None:
    if not results:
        print("No results found. Did you run embed_index.py first?")
        return

    print(f'Query: "{query_text}"')
    print(f"Mode: {mode}")
    confidence = assess_retrieval_confidence(results, mode=mode)
    confidence_label = "confident" if confidence["confident"] else "weak"
    print(f"Retrieval confidence: {confidence_label} ({confidence['reason']})")
    print(f"Top {len(results)} result(s):")
    print("-" * 60)

    for result in results:
        metadata = result["metadata"]
        distance = result["distance"]
        similarity = result["vector_score"]

        print(
            f"\n#{result['rank']} | distance={distance:.4f} "
            f"| similarity~{similarity:.4f} | id={result['id']}"
        )
        if mode in {"hybrid", "hybrid_mmr"}:
            print(
                f"   hybrid={result['hybrid_score']:.4f} "
                f"| bm25={result['bm25_score']:.4f} "
                f"| keyword={result['keyword_score']:.4f} "
                f"| heading={result['heading_score']:.4f} "
                f"| credibility={result['credibility_score_normalized']:.4f}"
            )
            if mode == "hybrid_mmr":
                print(
                    f"   mmr={result.get('mmr_score', 0.0):.4f} "
                    f"| redundancy_penalty={result.get('mmr_diversity_penalty', 0.0):.4f}"
                )
        print(
            f"   source: {metadata.get('source_file')} "
            f"| pages: {metadata.get('start_page')}-{metadata.get('end_page')} "
            f"| heading: {metadata.get('section_heading')}"
        )
        print(
            f"   document_id: {metadata.get('document_id', 'unknown')} "
            f"| path: {metadata.get('source_path', 'unknown')}"
        )
        print(
            f"   subject: {metadata.get('subject', 'unknown')} "
            f"| type: {metadata.get('doc_type', 'unknown')} "
            f"| source_type: {metadata.get('source_type', 'unknown')} "
            f"| credibility: {metadata.get('credibility_label', 'unknown')} "
            f"({metadata.get('credibility_score', 0)}/100)"
        )
        print(f"   text: {_preview(result['document'])}")


def main() -> None:
    _configure_stdout()

    parser = argparse.ArgumentParser(
        description="Semantic/hybrid search over the local scholAR index."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Natural-language search query.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return. Default: {DEFAULT_TOP_K}.",
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
    args = parser.parse_args()

    query_text = " ".join(args.query).strip()
    if not query_text:
        query_text = input("Enter a search query: ").strip()

    collection = get_chroma_collection()
    print(f"Total chunks currently in Chroma: {collection.count()}\n")

    metadata_filters = build_metadata_filters(
        subject=args.subject,
        doc_type=args.doc_type,
        source_type=args.source_type,
        min_credibility=args.min_credibility,
    )

    results = search_chunks(
        query_text=query_text,
        top_k=args.top_k,
        source_file=args.source,
        mode=args.mode,
        metadata_filters=metadata_filters,
    )
    print_results(query_text, results, args.top_k, args.mode)


if __name__ == "__main__":
    main()
