"""Splits filing text into overlapping chunks ready for embedding.

Two-level, structure-aware chunking (replacing Phase 5's original fixed-window
chunking):

1. Section split: deep_dive/section_splitter.py's split_sections() divides the
   filing into Item-N sections first (falling back to roughly-equal, sentence-
   snapped parts for non-standard filings) — the same split Phase 7 already used
   for deep-dive indexing, now shared by the main index too. A chunk never
   crosses a section boundary.
2. Sentence-safe sub-chunking: within each section, chunk_text() packs whole
   sentences into ~CHUNK_SIZE_CHARS groups with a sentence of overlap between
   consecutive chunks, so a chunk never splits a sentence across a boundary —
   only a single pathological sentence longer than the chunk size itself falls
   back to raw word-boundary splitting.

Chunk size is tuned to the embedder's model: all-MiniLM-L6-v2 truncates past
256 tokens (~1000 chars of English prose), so chunks stay under that.

Real bug this replaces (see progress.md's Phase 5 write-up): the original
1000-char/200-overlap windows were cut by raw character count and snapped only
to the nearest word boundary, so a sentence could land right at a chunk edge —
confirmed via two retrieval test misses whose sampled "question" snippet
started or ended mid-word. Overlap also duplicated whichever sentence happened
to sit in the overlap zone into two chunks with no record of that, so a
retriever returning the "other" valid copy of a sentence scored as a miss.
Packing whole sentences (never splitting one across a boundary) removes the
ambiguity at the source instead of patching the test to tolerate either copy.
"""

import re

CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
OVERLAP_SENTENCES = 1

# Splits on sentence-ending punctuation followed by whitespace and a capital
# letter/digit/quote — good enough for get_filing's already-flattened (single-
# spaced, no paragraph breaks) filing text; doesn't try to handle abbreviations
# perfectly, since an occasional over-split sentence is harmless here (it just
# becomes two small "sentences" packed back together by chunk_text anyway).
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"‘“])')


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _hard_split(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Fallback for a single sentence longer than chunk_size (e.g. a flattened
    table row with no sentence punctuation) — same word-boundary window
    splitting Phase 5 originally used for everything."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
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


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[str]:
    """Splits `text` into chunks of whole sentences, each up to ~chunk_size
    chars, carrying `overlap_sentences` sentences of overlap between
    consecutive chunks for continuity. Never splits a sentence across two
    chunks, except a single sentence longer than chunk_size on its own, which
    falls back to word-boundary splitting (_hard_split)."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush():
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        if len(sentence) > chunk_size:
            _flush()
            current.clear()
            current_len = 0
            chunks.extend(_hard_split(sentence, chunk_size))
            continue

        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            _flush()
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            overlap_len = sum(len(s) for s in overlap) + max(0, len(overlap) - 1)
            # An overlap sentence can itself be up to chunk_size long (anything
            # bigger already went through _hard_split above and never lands in
            # `current`) — if seeding the next chunk with it would already leave
            # no room for the sentence that triggered this flush, drop the
            # overlap for this one boundary instead of producing a chunk that
            # can run up to ~2x chunk_size and silently lose its tail past the
            # embedder's ~256-token truncation. Continuity is a nice-to-have;
            # staying within chunk_size is not optional.
            if overlap_len + len(sentence) + 1 <= chunk_size:
                current = overlap
                current_len = overlap_len
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    _flush()
    return chunks


def chunk_filing(filing_data: dict, section: str | None = None) -> list[dict]:
    """Chunks a get_filing() `data` dict into records ready for rag.chroma_store.

    Returns a list of {"id", "text", "metadata"} dicts. metadata carries company,
    filing_type, filed_date, source_url, section, and chunk_index (restarting at
    0 within each section).

    If `section` is given, `filing_data["raw_text"]` is treated as already
    scoped to that one section (deep_dive/section_splitter.py's call pattern) —
    only level-2 sentence-safe sub-chunking runs. If `section` is None (the
    main-index call pattern), this does the level-1 section split itself first
    (deep_dive.section_splitter.split_sections, imported locally to avoid a
    module-level import cycle with that module's own `from rag.chunker import
    chunk_filing`), then sentence-safe sub-chunks within each section — so a
    chunk never crosses a section boundary and metadata["section"] carries the
    real Item-N label instead of always being blank.

    Ids are deterministic (company + filing_type + section + chunk_index, no
    random component) so re-indexing the same filing overwrites the same
    Chroma rows via upsert instead of accumulating duplicates alongside stale
    data.
    """
    company = filing_data["company"]
    filing_type = filing_data["filing_type"]

    if section is not None:
        sub_sections = [{"section": section, "text": filing_data["raw_text"]}]
    else:
        from deep_dive.section_splitter import split_sections

        sub_sections = split_sections(filing_data["raw_text"])

    records = []
    for sub in sub_sections:
        label = sub["section"]
        for i, text in enumerate(chunk_text(sub["text"])):
            records.append(
                {
                    "id": f"{company}-{filing_type}-{label or 'full'}-{i}",
                    "text": text,
                    "metadata": {
                        "company": company,
                        "filing_type": filing_type,
                        "filed_date": filing_data["filed_date"],
                        "source_url": filing_data["source_url"],
                        "section": label,
                        "chunk_index": i,
                    },
                }
            )
    return records
