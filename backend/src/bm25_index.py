"""
bm25_index.py

Small dependency-free BM25 lexical index for hybrid retrieval.

The vector index lives in Chroma. This file builds a separate lexical index
from the same Chroma chunk ids/documents/metadata, so vector and BM25 results
can be merged by stable chunk id.
"""

import math
import os
import pickle
import re
from collections import Counter

BM25_INDEX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "db",
    "bm25_index.pkl",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    """
    Tokenizer used by BM25 and lightweight keyword scoring.
    """
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _bm25_tokens(document: str, metadata: dict) -> list[str]:
    """
    Tokens used for BM25 lexical retrieval.

    Headings are repeated to give topic titles more lexical weight without
    needing a separate fielded search engine.
    """
    heading = metadata.get("section_heading") or ""
    source_file = metadata.get("source_file") or ""
    return tokenize(f"{heading} {heading} {source_file} {document}")


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {}
    max_score = max(raw_scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in raw_scores}
    return {key: value / max_score for key, value in raw_scores.items()}


def records_from_chroma(collection) -> list[dict]:
    """
    Read chunk ids/documents/metadata from Chroma.
    """
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", [])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    records = []
    for index, chunk_id in enumerate(ids):
        records.append({
            "id": chunk_id,
            "document": documents[index],
            "metadata": metadatas[index],
        })
    return records


def build_bm25_index(records: list[dict]) -> dict:
    """
    Build a serializable BM25 index from chunk records.
    """
    tokenized_documents = []
    document_frequencies = Counter()

    for record in records:
        tokens = _bm25_tokens(record["document"], record["metadata"])
        tokenized_documents.append(tokens)
        document_frequencies.update(set(tokens))

    doc_count = len(records)
    average_document_length = (
        sum(len(tokens) for tokens in tokenized_documents) / doc_count
        if doc_count
        else 0
    )

    return {
        "version": "bm25_v1",
        "records": records,
        "tokenized_documents": tokenized_documents,
        "document_frequencies": dict(document_frequencies),
        "average_document_length": average_document_length,
        "document_count": doc_count,
        "k1": 1.5,
        "b": 0.75,
    }


def save_bm25_index(index: dict, path: str = BM25_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(index, file)


def delete_bm25_index(path: str = BM25_INDEX_PATH) -> None:
    """
    Remove the stored BM25 index if it exists.

    Used during --rebuild so the lexical index cannot stay stale if Chroma is
    being reconstructed.
    """
    if os.path.exists(path):
        os.remove(path)


def load_bm25_index(path: str = BM25_INDEX_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def build_and_save_bm25_index(collection, path: str = BM25_INDEX_PATH) -> dict:
    records = records_from_chroma(collection)
    index = build_bm25_index(records)
    save_bm25_index(index, path)
    return index


def load_or_build_bm25_index(collection, path: str = BM25_INDEX_PATH) -> dict:
    """
    Prefer the stored BM25 index. If missing, build it from Chroma as a safe
    fallback so search still works.
    """
    index = load_bm25_index(path)
    if index is not None:
        return index
    return build_bm25_index(records_from_chroma(collection))


def score_bm25(
    query_text: str,
    index: dict,
    source_file: str | None = None,
    metadata_filters: dict | None = None,
) -> dict[str, float]:
    """
    Score query against a stored BM25 index.
    Returns normalized scores keyed by chunk id.
    """
    query_terms = tokenize(query_text)
    if not query_terms or not index.get("records"):
        return {}

    records = index["records"]
    tokenized_documents = index["tokenized_documents"]
    document_frequencies = index["document_frequencies"]
    doc_count = index["document_count"]
    average_document_length = index["average_document_length"]
    k1 = index.get("k1", 1.5)
    b = index.get("b", 0.75)

    if average_document_length == 0:
        return {}

    raw_scores = {}

    metadata_filters = metadata_filters or {}

    for record, tokens in zip(records, tokenized_documents):
        if source_file and record["metadata"].get("source_file") != source_file:
            continue
        if not _metadata_matches(record["metadata"], metadata_filters):
            continue

        if not tokens:
            raw_scores[record["id"]] = 0.0
            continue

        term_counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0

        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue

            docs_with_term = document_frequencies.get(term, 0)
            idf = math.log(
                1 + ((doc_count - docs_with_term + 0.5) / (docs_with_term + 0.5))
            )
            denominator = term_frequency + k1 * (
                1 - b + b * (doc_len / average_document_length)
            )
            score += idf * ((term_frequency * (k1 + 1)) / denominator)

        raw_scores[record["id"]] = score

    return _normalize_scores(raw_scores)


def _metadata_matches(metadata: dict, filters: dict) -> bool:
    """
    Apply lightweight metadata filters for BM25 candidates.

    Chroma applies the same filters for vector candidates. BM25 needs its own
    check because it is a separate local lexical index.
    """
    subject = filters.get("subject")
    if subject and metadata.get("subject") != subject:
        return False

    doc_type = filters.get("doc_type")
    if doc_type and metadata.get("doc_type") != doc_type:
        return False

    source_type = filters.get("source_type")
    if source_type and metadata.get("source_type") != source_type:
        return False

    min_credibility = filters.get("min_credibility")
    if min_credibility is not None:
        if (metadata.get("credibility_score") or 0) < min_credibility:
            return False

    return True


def records_by_id(index: dict) -> dict[str, dict]:
    return {record["id"]: record for record in index.get("records", [])}
