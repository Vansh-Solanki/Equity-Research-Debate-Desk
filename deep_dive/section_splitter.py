r"""Splits a 10-K's flattened text into per-Item sections using SEC's standard
Item-N header structure (spec.md section 7), for deep-dive retrieval filtering.

Real finding while building this: `mcp_server/tools/get_filing.py`'s `_strip_html`
collapses all whitespace to single spaces (`re.sub(r"\s+", " ", text)`), so
`raw_text` has no paragraph breaks to split on — the spec's literal
"paragraph-count-based chunking" fallback has nothing to count. `_fallback_split`
instead splits into FALLBACK_PARTS roughly equal character-length parts, snapped
to sentence boundaries (same word/sentence-boundary-snapping spirit as
rag/chunker.py) so a part never starts or ends mid-sentence.

Second finding, verified against a real AAPL 10-K: every 10-K's front matter
includes a table of contents that lists every Item header ("Item 1A. Risk
Factors 5 Item 1B. ...") right next to its page number — which matches the same
header regex as the real section later in the document. Naively splitting at
every match would carve the TOC itself into a dozen near-empty pseudo-sections
before the real ones.

First fix attempted — keep only the occurrence of each item number with the
largest gap to the next distinct header — was itself a real bug: for items
whose real body content is trivially short ("Item 16. Form 10-K Summary None."
right before the signature block; "Item 11. Executive Compensation" deferred
entirely to the proxy statement), the real section's own trailing gap can be
*smaller* than the TOC entry's trailing gap (the TOC's last row is often
followed by a page or two of legal preamble before Item 1 actually starts,
which reads as a bigger gap than a two-sentence real "Item 16" section
followed immediately by the next item). Verified directly on AAPL's 10-K: TOC's
"Item 16." had a 2052-char trailing gap vs. the real section's 2025-char gap —
close enough that the largest-gap heuristic picked the TOC line.

Fixed properly by detecting the TOC as a structural feature — a dense *run* of
matches near the start of the document where each is within TOC_GAP_THRESHOLD
chars of the previous one (confirmed on AAPL: every TOC-internal gap was under
230 chars, and the jump from the TOC's last entry to the real "Item 1." body
was 2052 chars) — and excluding that whole run outright, rather than scoring
each occurrence's own trailing gap. Whatever remains after the TOC run is
assumed to be real body headers; duplicates there (e.g. a later cross-reference
that happens to match the header pattern) keep the first occurrence, since the
real section always appears before any later mention of it.

Third real bug, found testing MSFT (not present in AAPL): MSFT's TOC uses
Title Case ("Item 1A. Risk Factors") but its real body section headers are
ALL CAPS ("ITEM 1A. RIS K FACTORS" — note filer templates sometimes even
insert a stray space mid-word from a span-tag boundary that get_filing's HTML
stripping collapses; irrelevant here since only the header prefix is matched,
not the title text). The original regex only matched literal "Item", so every
one of MSFT's real body headers was silently invisible to it — every detected
match was either a TOC row or a "PART I Item 1"-style running page header with
no period (already excluded by requiring "). \s+[A-Z]" right after the
number), leaving only the TOC matches to work with and producing badly wrong
splits. Fixed with an explicit `(?:Item|ITEM)` alternation (not a blanket
re.IGNORECASE, which would also weaken the "next char is uppercase" check used
to confirm a real heading rather than a lowercase in-sentence mention).
Verified against both AAPL (22 sections, all correctly ordered and non-trivial
sized) and MSFT (22 sections, all pulling from real body content) after this
fix.
"""

import re

from mcp_server.tools import get_filing as _get_filing
from rag.chroma_store import delete_filing_chunks, get_or_create_collection, upsert_chunks
from rag.chunker import chunk_filing
from rag.embedder import embed_texts

_ITEM_HEADER_RE = re.compile(r"(?:Item|ITEM)\s+(\d{1,2}[A-Z]?)\.\s+(?=[A-Z])")

# Standard 10-K item structure (spec.md section 7's short list, extended to the
# full standard set) — used only to build human-readable labels; an item number
# not in this map still gets split correctly, just labeled generically.
_KNOWN_ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

MIN_STANDARD_HEADERS = 3
FALLBACK_PARTS = 4

# Matches consecutive TOC row spacing observed on a real AAPL 10-K (every
# TOC-internal gap was under 230 chars); comfortably below the 2000+ char gap
# from the TOC's last row to the real Item 1 body seen on the same filing.
TOC_GAP_THRESHOLD = 300
# A short filing could plausibly have two real headers this close together;
# require a longer unbroken run before treating it as a TOC rather than
# coincidence — a real 10-K's TOC lists well over this many items.
MIN_TOC_RUN_LENGTH = 4


def _label(item_number: str) -> str:
    title = _KNOWN_ITEM_TITLES.get(item_number)
    return f"Item {item_number} — {title}" if title else f"Item {item_number}"


def _find_section_headers(text: str) -> list[tuple[int, str]]:
    """Returns [(start_pos, item_number), ...] for real section headers, sorted by
    position, with the table-of-contents listing excluded (see module docstring)."""
    raw_matches = [(m.start(), m.group(1)) for m in _ITEM_HEADER_RE.finditer(text)]
    if not raw_matches:
        return []

    # The TOC is a dense run of matches starting at the very first match, each
    # within TOC_GAP_THRESHOLD chars of the previous one — detect its extent by
    # walking forward until a gap breaks the run, rather than judging each
    # occurrence's own trailing gap (see module docstring for why that's unsound).
    run_end = 0
    for i in range(1, len(raw_matches)):
        if raw_matches[i][0] - raw_matches[i - 1][0] < TOC_GAP_THRESHOLD:
            run_end = i
        else:
            break

    if run_end + 1 >= MIN_TOC_RUN_LENGTH:
        body_matches = raw_matches[run_end + 1 :]
    else:
        body_matches = raw_matches

    if len(body_matches) < MIN_STANDARD_HEADERS:
        # Safety net: excluding the detected "TOC" left too little to work with
        # (e.g. a real header was misdetected as part of the run) — fall back to
        # every raw match rather than losing real sections.
        body_matches = raw_matches

    # Keep the first occurrence of each item number: the real section always
    # appears before any later cross-reference that happens to match the header
    # pattern.
    seen: set[str] = set()
    deduped: list[tuple[int, str]] = []
    for pos, number in body_matches:
        if number in seen:
            continue
        seen.add(number)
        deduped.append((pos, number))

    return sorted(deduped, key=lambda pair: pair[0])


# Matches a real sentence end (period/!/?) followed by whitespace and an
# uppercase letter/digit/quote — same shape as rag/chunker.py's
# _SENTENCE_SPLIT_RE. A plain `text.rfind(". ", ...)` originally snapped this
# window boundary, which doesn't distinguish a real sentence end from ". "
# inside an abbreviation ("Apple Inc. designs...", "the U.S. dollar...") —
# confirmed it could snap a part boundary right after "Inc." and hand
# rag.chunker.chunk_text() a fallback "section" that starts mid-sentence
# ("designs and sells products..."), the exact mid-sentence-boundary bug this
# whole two-level chunking scheme exists to prevent. Only reachable when a
# filing has fewer than MIN_STANDARD_HEADERS real Item headers (this fallback
# path), but now that rag.chunker.chunk_filing() calls split_sections() for
# every main-index filing too, not just deep-dive's optional path, a
# non-standard filing would have hit this silently.
_SENTENCE_BOUNDARY_RE = re.compile(r'[.!?]\s+(?=[A-Z0-9"‘“])')


def _last_sentence_boundary(text: str, start: int, end: int) -> int | None:
    """Returns the position right after the last real sentence-ending
    punctuation within text[start:end], or None if none is found."""
    matches = list(_SENTENCE_BOUNDARY_RE.finditer(text, start, end))
    return matches[-1].start() + 1 if matches else None


def _fallback_split(text: str) -> list[dict]:
    """No paragraph breaks survive get_filing's whitespace collapsing (see module
    docstring), so this splits into FALLBACK_PARTS roughly equal character-length
    windows, snapped to the nearest real sentence boundary (_last_sentence_boundary)
    so a part doesn't start or end mid-sentence."""
    text = text.strip()
    if not text:
        return []

    part_size = max(1, len(text) // FALLBACK_PARTS)
    sections = []
    start = 0
    for i in range(FALLBACK_PARTS):
        if start >= len(text):
            break
        end = start + part_size if i < FALLBACK_PARTS - 1 else len(text)
        if end < len(text):
            snap = _last_sentence_boundary(text, start, end)
            if snap is not None and snap > start:
                end = snap
        part_text = text[start:end].strip()
        if part_text:
            sections.append({"section": f"Part {i + 1}", "text": part_text})
        start = end
    return sections


def split_sections(raw_text: str) -> list[dict]:
    """Splits `raw_text` (a get_filing() raw_text string) into per-section records:
    [{"section": label, "text": section_text}, ...], in document order.

    Uses SEC's standard Item-N header structure; if fewer than
    MIN_STANDARD_HEADERS real headers are found (a non-standard or heavily
    templated filing), falls back to _fallback_split rather than failing —
    per spec.md section 7's fallback requirement.
    """
    headers = _find_section_headers(raw_text)
    if len(headers) < MIN_STANDARD_HEADERS:
        return _fallback_split(raw_text)

    sections = []
    for i, (pos, number) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(raw_text)
        text = raw_text[pos:end].strip()
        if text:
            sections.append({"section": _label(number), "text": text})
    return sections


def _indexed_section_labels(company: str, filing_type: str) -> set[str]:
    """Real (non-blank) section labels already indexed in `company`'s collection
    for `filing_type` — used by index_filing_sections to skip redundant
    re-embedding when rag.retriever.index_company_filing already covered them."""
    collection = get_or_create_collection(company)
    if collection.count() == 0:
        return set()
    existing = collection.get(where={"filing_type": filing_type}, include=["metadatas"])
    return {m["section"] for m in existing["metadatas"] if m.get("section")}


def index_filing_sections(company: str, filing_type: str = "10-K") -> dict:
    """Fetches `company`'s filing, splits it into sections (split_sections), and
    chunks + indexes each section separately in Chroma with metadata["section"]
    set to that section's label — reusing rag.chunker.chunk_filing's existing
    `section` parameter, so rag.retriever.retrieve(..., where={"section": label})
    can later scope retrieval to one section only.

    This indexes into the same per-company collection rag.retriever.index_company_filing
    uses for the whole filing — the two coexist without clobbering each other
    since chroma_store.delete_filing_chunks scopes deletes to (filing_type, section).

    rag.chunker.chunk_filing(section=None) now section-splits internally too (see
    rag/chunker.py's Phase 5 rework), so index_company_filing already produces
    byte-identical chunks (same ids, same text) for every real section — calling
    this function after a debate has already indexed the filing would otherwise
    re-fetch, re-chunk, and re-embed the entire filing a second time for rows
    that upsert would just overwrite with identical content. If every section
    this filing actually has is already indexed, skip the chunk/embed/upsert
    work entirely and just return the (still freshly split, so still correctly
    ordered) label list.

    Returns {"success": bool, "sections": [str, ...], "error": str | None} — the
    section labels list is what Phase 9's UI generates one deep-dive button per.
    """
    result = _get_filing(company, filing_type)
    if not result["success"]:
        return {"success": False, "sections": [], "error": result["error"]}

    filing_data = result["data"]
    sections = split_sections(filing_data["raw_text"])
    if not sections:
        return {"success": False, "sections": [], "error": "filing produced no sections"}

    all_labels = [s["section"] for s in sections]
    if set(all_labels) <= _indexed_section_labels(company, filing_type):
        return {"success": True, "sections": all_labels, "error": None}

    labels = []
    for section in sections:
        label = section["section"]
        section_filing_data = {**filing_data, "raw_text": section["text"]}
        records = chunk_filing(section_filing_data, section=label)
        if not records:
            continue
        delete_filing_chunks(company, filing_type, section=label)
        embeddings = embed_texts([r["text"] for r in records])
        upsert_chunks(company, records, embeddings)
        labels.append(label)

    return {"success": True, "sections": labels, "error": None}
