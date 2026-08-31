"""Splits filing text into overlapping chunks ready for embedding.

Generic fixed-window chunking with overlap — not the Item-N section-aware split
(that's Phase 7's deep_dive/section_splitter.py, applied separately for deep-dive
retrieval filtering). Chunk size is tuned to the embedder's model: all-MiniLM-L6-v2
truncates past 256 tokens (~1000 chars of English prose), so chunks stay under that.
"""

CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Splits `text` into overlapping windows, snapped to word boundaries so chunks
    don't cut off mid-word."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # snap to the last word boundary within the window so we don't split a word
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def chunk_filing(filing_data: dict, section: str | None = None) -> list[dict]:
    """Chunks a get_filing() `data` dict into records ready for rag.chroma_store.

    Returns a list of {"id", "text", "metadata"} dicts. metadata carries company,
    filing_type, filed_date, source_url, section (None here; Phase 7's deep-dive
    indexing passes the actual Item-N section name), and chunk_index. Ids are
    deterministic (company + filing_type + section + chunk_index, no random
    component) so re-indexing the same filing overwrites the same Chroma rows via
    upsert instead of accumulating duplicates alongside stale data.
    """
    company = filing_data["company"]
    filing_type = filing_data["filing_type"]
    texts = chunk_text(filing_data["raw_text"])

    records = []
    for i, text in enumerate(texts):
        records.append(
            {
                "id": f"{company}-{filing_type}-{section or 'full'}-{i}",
                "text": text,
                "metadata": {
                    "company": company,
                    "filing_type": filing_type,
                    "filed_date": filing_data["filed_date"],
                    "source_url": filing_data["source_url"],
                    "section": section or "",
                    "chunk_index": i,
                },
            }
        )
    return records
