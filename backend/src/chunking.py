"""
chunking.py

Takes the page-wise output of extract.py and splits it into chunks,
ready for embedding.

DESIGN: per-page chunking + explicit boundary merge.

Instead of joining the whole document into one string and then searching
to figure out which page each resulting chunk came from (fragile - requires
whitespace-normalization fallbacks, overlap-aware cursor tricks, etc.),
this version chunks each page independently first. Page numbers are known
by construction (we are literally looping over pages), so there is no
search/reconstruction step at all.

The tradeoff: a topic can be cut off right at a page boundary (e.g. 150
words on page 3, continuing for 450 more words on page 4). To handle this,
a separate, explicit merge pass looks at the last chunk of page N and the
first chunk of page N+1, and merges them if simple, inspectable signals
suggest the topic is continuing rather than starting fresh:
    - the next page's first chunk does NOT start with a heading
    - the current page's last chunk does NOT end with sentence-ending
      punctuation (i.e. looks cut off mid-thought)
    - the combined chunk stays below a word-count safety limit, so retrieval
      chunks do not become too broad

Output chunk shape:
{
    "chunk_id": "deadlocks_pdf_0003",
    "chunk_text": "...",
    "source_file": "Deadlocks.pdf",
    "start_page": 4,
    "end_page": 5,
    "pages": [4, 5],
    "section_heading": "Deadlock Prevention",
    "parent_headings": ["Deadlocks"],
    "image_count": 1,
    "text_word_count": 42,
    "should_embed": True,
    "merged_across_pages": True,
    "chunk_index": 3
}
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

IMAGE_OMITTED_MARKER = "intentionally omitted"
SENTENCE_END_CHARS = (".", "?", "!", ":", '"', "'")
MIN_WORDS_TO_EMBED = 8
MAX_WORDS_AFTER_PAGE_MERGE = 650


def _safe_chunk_id_source(source_file: str) -> str:
    return (
        source_file.lower()
        .replace(".", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _clean_page_text(text: str) -> tuple[str, int]:
    """
    Strips line-number artifacts and image-omitted placeholders.
    Returns (cleaned_text, image_count_removed).
    """
    image_count = text.count(IMAGE_OMITTED_MARKER)
    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            continue
        if IMAGE_OMITTED_MARKER in stripped:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip(), image_count


def _starts_with_heading(text: str) -> bool:
    return text.lstrip().startswith("#")


def _looks_complete(text: str) -> bool:
    """
    Heuristic: does this text look like it ends on a finished thought?
    Used to decide whether the next page's content is likely a continuation.
    """
    stripped = text.strip()
    if not stripped:
        return True  # nothing to continue
    return stripped.endswith(SENTENCE_END_CHARS)


def _count_words(text: str) -> int:
    return sum(
        1
        for token in text.replace("#", " ").replace("*", " ").split()
        if any(char.isalnum() for char in token)
    )


def _within_merge_word_limit(left_text: str, right_text: str) -> bool:
    """
    Avoid creating very large cross-page chunks.

    If two page-boundary chunks are both substantial, keeping them separate is
    usually better for retrieval precision. Search can still return both
    chunks as neighboring results because each keeps accurate page metadata.
    """
    combined_text = left_text.rstrip() + " " + right_text.lstrip()
    return _count_words(combined_text) <= MAX_WORDS_AFTER_PAGE_MERGE


def _heading_context(metadata: dict) -> tuple[str | None, list[str]]:
    headings = [metadata.get("h1"), metadata.get("h2"), metadata.get("h3")]
    headings = [h for h in headings if h]
    if not headings:
        return None, []
    return headings[-1], headings[:-1]


def _split_single_page(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Splits ONE page's text into chunks. Tries markdown headings first,
    falls back to plain recursive splitting if no headings are found.
    No document-wide joining, no position search - operates purely on
    this one page's text.

    Returns a list of {"text", "section_heading", "parent_headings"}.
    """
    if not text.strip():
        return []

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )

    try:
        sections = header_splitter.split_text(text)
    except Exception:
        sections = []

    if not sections:
        return [
            {"text": piece.strip(), "section_heading": None, "parent_headings": []}
            for piece in recursive_splitter.split_text(text)
            if piece.strip()
        ]

    results = []
    for section in sections:
        section_text = section.page_content.strip()
        if not section_text:
            continue
        section_heading, parent_headings = _heading_context(section.metadata)

        if len(section_text) > chunk_size:
            for piece in recursive_splitter.split_text(section_text):
                piece = piece.strip()
                if piece:
                    results.append({
                        "text": piece,
                        "section_heading": section_heading,
                        "parent_headings": parent_headings,
                    })
        else:
            results.append({
                "text": section_text,
                "section_heading": section_heading,
                "parent_headings": parent_headings,
            })

    return results


def chunk_pages(
    pages: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list:
    """
    Main chunking entry point.

    pages: list of {"page_number", "source_file", "text"} from extract.py
    Returns: list of chunk dicts (see module docstring for shape)
    """
    if not pages:
        return []

    source_file = next(
        (p.get("source_file") for p in pages if p.get("source_file")),
        "unknown_source",
    )

    # Step 1: chunk every page independently. Page number is known by
    # construction - no search required.
    raw_chunks = []  # each item: dict with page_number attached
    for page in pages:
        page_number = page.get("page_number")
        raw_text = page.get("text", "")
        cleaned_text, image_count = _clean_page_text(raw_text)

        page_pieces = _split_single_page(cleaned_text, chunk_size, chunk_overlap)

        for piece_index, piece in enumerate(page_pieces):
            raw_chunks.append({
                "text": piece["text"],
                "section_heading": piece["section_heading"],
                "parent_headings": piece["parent_headings"],
                "start_page": page_number,
                "end_page": page_number,
                "pages": [page_number],
                # attribute the page's image count to its first chunk only,
                # so a page with multiple chunks doesn't multiply-count images
                "image_count": image_count if piece_index == 0 else 0,
            })

    # Step 2: explicit merge pass across page boundaries.
    # Walk through chunks; merge the last chunk of page N into the first
    # chunk of page N+1 when it looks like a continuation, not a new topic.
    merged_chunks = []
    i = 0
    while i < len(raw_chunks):
        current = raw_chunks[i]

        if merged_chunks:
            previous = merged_chunks[-1]
            is_new_page = current["start_page"] != previous["end_page"]
            continues_topic = (
                is_new_page
                and not _starts_with_heading(current["text"])
                and not _looks_complete(previous["text"])
                and _within_merge_word_limit(previous["text"], current["text"])
            )
            if continues_topic:
                previous["text"] = previous["text"].rstrip() + " " + current["text"].lstrip()
                previous["end_page"] = current["end_page"]
                previous["pages"] = sorted(set(previous["pages"] + current["pages"]))
                previous["image_count"] += current["image_count"]
                previous["merged_across_pages"] = True
                # keep the earlier section heading if the later chunk had none
                if not previous["section_heading"] and current["section_heading"]:
                    previous["section_heading"] = current["section_heading"]
                    previous["parent_headings"] = current["parent_headings"]
                i += 1
                continue

        merged_chunks.append({**current, "merged_across_pages": False})
        i += 1

    # Step 3: attach final metadata (chunk_id, word count, should_embed)
    final_chunks = []
    for index, chunk in enumerate(merged_chunks):
        text_word_count = _count_words(chunk["text"])
        final_chunks.append({
            "chunk_id": f"{_safe_chunk_id_source(source_file)}_{index:04d}",
            "chunk_text": chunk["text"].strip(),
            "source_file": source_file,
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "pages": chunk["pages"],
            "section_heading": chunk["section_heading"],
            "parent_headings": chunk["parent_headings"],
            "image_count": chunk["image_count"],
            "text_word_count": text_word_count,
            "should_embed": text_word_count >= MIN_WORDS_TO_EMBED,
            "merged_across_pages": chunk["merged_across_pages"],
            "chunk_index": index,
        })

    return final_chunks


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sys.path.insert(0, ".")
    from extract import extract_text

    if len(sys.argv) < 2:
        print("Usage: python chunking.py <pdf_path> [pdf_path ...]")
        sys.exit(1)

    for pdf_path in sys.argv[1:]:
        output = extract_text(pdf_path)
        chunks = chunk_pages(output["pages"])

        print("\n========================")
        print(f"File: {output['filename']} | Pages: {output['page_count']}")
        print("========================")
        print(f"Total chunks: {len(chunks)}")
        merged_count = sum(1 for c in chunks if c["merged_across_pages"])
        print(f"Chunks merged across a page boundary: {merged_count}")

        print("\n--- All chunks ---")
        for chunk in chunks:
            heading = chunk["section_heading"] or "no heading"
            merge_flag = " [MERGED]" if chunk["merged_across_pages"] else ""
            print(
                f"\n[pages {chunk['start_page']}-{chunk['end_page']} | {heading} | "
                f"words: {chunk['text_word_count']} | embed: {chunk['should_embed']}]"
                f"{merge_flag}"
            )
            print(chunk["chunk_text"][:])
