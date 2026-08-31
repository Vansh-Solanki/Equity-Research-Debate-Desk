# Equity Research Debate Desk — progress

**Status:** Phase 5 complete (Chroma RAG index: chunk -> embed -> store -> rerank), Phase 6 pending
**Last updated:** August 30, 2026

## One-line summary

Two AI analyst agents (Bull and Bear) debate a company's real SEC filing using retrieved, fact-checked evidence, a Judge agent scores the debate and writes a memo, and an optional deep-dive mode spawns section-specific sub-agents on demand. Built entirely on free tools, no paid subscriptions.

## Architecture (locked in)

```
Data sources (SEC EDGAR, yfinance, news RSS)
        |
MCP tool layer (get_filing, get_price, get_news)
        |
RAG index (chunk -> embed -> Chroma -> cross-encoder rerank)
        |
Agent crew (Bull, Bear, Judge — CrewAI + Groq)
        |-- Deep-dive spawner (sub-agent per filing section, on demand)
        |
Output layer (memo, Streamlit UI, Langfuse/local logging)
```

Deployment layer (added later, also locked in):
```
Browser -> Streamlit frontend -> FastAPI backend -> Redis job queue -> Worker pool
(worker pool runs the agent crew, calls Groq + Chroma, writes results back)
Hosted as a Docker container on Hugging Face Spaces (free)
```

## Tech stack (all free tier / open source)

| Layer | Tool | Notes |
|---|---|---|
| LLM | Groq API (free tier) | Confirmed as primary driver |
| Agent framework | CrewAI | Defines Bull, Bear, Judge roles |
| Tool access | Self-built MCP server | Wraps EDGAR, yfinance, news as callable tools |
| Filings | SEC EDGAR public API | Free, no key required |
| Price/financial data | yfinance | Free Python package |
| News | Free RSS feeds | Catches recent events not yet in filings |
| Embeddings | sentence-transformers | Local, no API cost |
| Vector store | **ChromaDB** | Chosen over FAISS, Qdrant, LanceDB, etc. — best fit for single-filing scale + metadata filtering for deep-dive sections |
| Reranking | cross-encoder (ms-marco-MiniLM) | Improves fact-check precision over raw vector similarity |
| Hallucination check | NLI entailment model (local, free) | Claim-level groundedness scoring, applied to all agents equally |
| Frontend | Streamlit | Debate feed UI + deep-dive buttons + memo card |
| Backend | FastAPI | Async, handles concurrent requests |
| Job queue | Redis (Upstash free tier) | Decouples request submission from long-running debate jobs |
| Observability | Langfuse (self-hosted or free tier) | Logs every tool call, retrieval, and judge decision |
| Hosting | Hugging Face Spaces (Docker) | Free, credible AI hosting choice |
| CI/CD | GitHub Actions | Free for public repos |

## Concept coverage (against the 13-topic checklist)

| Topic | Covered? | Where |
|---|---|---|
| MCP | Yes | Tool layer wrapping EDGAR/yfinance/news |
| A2A | No | Not part of this project — planned for the second (general) project |
| LangGraph | Optional | Not required by current design |
| CrewAI | Yes | Bull/Bear/Judge roles |
| Claude Agent SDK | No | Planned for the second (general) project |
| Short-term memory | Partial | Per-session debate history; not a deep memory system |
| Context engineering | Yes | Chunking + RAG for long filings |
| Orchestrator-worker | Yes | Deep-dive spawner dispatches section sub-agents |
| Debate/adversarial review | Yes | Core mechanic of the whole project |
| Hierarchical sub-agent spawning | Yes | One sub-agent per filing section, on demand |
| Evaluation/observability | Yes | Claim-level groundedness scoring, Judge F1, debate quality, retrieval recall, latency |
| Safety/guardrails | Yes | Entailment-based fact-checking on every agent's claims |
| Cloud/serverless deployment | Yes | Docker on Hugging Face Spaces, queue-worker pattern |

**Gap:** A2A and Claude Agent SDK are intentionally deferred to the second (general-domain) project.

## Key design decisions log

- **Debate turn order:** Fixed (Bull always first), implemented as an ordered list (`speaking_order = [bull_agent, bear_agent]`) so randomizing later is a one-line change if ever wanted. All logging/labeling is by agent name, never by position.
- **Independent research:** Bull and Bear run separate retrieval queries and write opening statements without seeing each other's statement. From round 2 onward, each agent reads the full transcript and must directly engage with the other's specific points.
- **Deep-dive trigger:** User-initiated only, via one button per filing section (dynamically generated from however many sections that specific filing has), not automatic. Clicking a button spawns exactly one sub-agent for that section; already-dived sections are cached, not rerun.
- **Hallucination checking:** One shared evaluation pipeline (claim extraction -> retrieval -> cross-encoder rerank -> NLI entailment check) applied uniformly to Bull, Bear, the Judge, and deep-dive sub-agents. No agent is exempt or self-certifying.
- **Scope boundary:** Confirmed this project does NOT cover merger and acquisition analysis (single-company research only, not deal/comparison scenarios).
- **Vector DB:** Chroma chosen over FAISS/Qdrant/Weaviate/Milvus/LanceDB/pgvector/Annoy/Redis-vector — best balance of ease of use, metadata filtering, and appropriate scale for this project.
- **Deployment concurrency:** Queue-worker pattern (Redis + worker pool) chosen over a naive single-process Streamlit app specifically to support multiple concurrent users without blocking or session collisions. Acknowledged limitation: real throughput is capped by Groq's free-tier rate limit, not by the architecture itself.

- **Groq model choice:** `llama-3.1-8b-instant` (originally assumed in spec.md) is no longer available on the free-tier account as of Aug 2026. Switched to `openai/gpt-oss-20b`, confirmed working with 1000 requests/day and 8000 tokens/min free-tier limits. Re-check `client.models.list()` if agents start hitting 404s later — Groq's free model lineup changes.

- **MCP SDK version:** Installing `crewai` in Phase 3 pinned `mcp` back down to v1.28.1 (from the v2.1.1 that Phase 2 was built against), which removed `mcp.server.mcpserver.MCPServer`. `mcp_server/server.py` now uses `from mcp.server.fastmcp import FastMCP` instead — decorator usage (`@mcp.tool()`) is unchanged. If `mcp` gets upgraded again later, re-check this import.
- **News source:** Spec suggested free RSS feeds generically; implemented against Google News RSS (`news.google.com/rss/search`) since it needs no API key and works for arbitrary company names, not just tickers.
- **SEC EDGAR user agent:** Set to `Equity Research Debate Desk vsolanki4565@gmail.com` in `.env` — SEC blocks requests with generic/placeholder user agents.
- **Filing text extraction:** Added `beautifulsoup4` (not in original spec's requirements.txt draft) to strip HTML from 10-K documents into plain text for later chunking in Phase 5. Raw text capped at 3M chars as a safety limit.

- **Phase 3 — CrewAI version and Groq quirks (crewai 1.15.18):**
  - `crewai`'s generic litellm-fallback `LLM` class (used for any model without a native provider integration, which includes Groq) stamps every message with a `cache_breakpoint` field intended for Anthropic-style prompt caching, but never strips it for providers that don't understand it. Groq's OpenAI-compatible endpoint strictly rejects unknown message fields and 400s on every call. Worked around in `agents/llm.py` by monkeypatching `crewai.llms.cache.mark_cache_breakpoint` to a no-op — both crewai executors re-import that name locally on each call, so patching the source module's attribute is enough. Revisit if crewai ships a real fix upstream.
  - `crewai` requires `litellm` installed separately as of 1.15.18 (`pip install litellm`) for any model that isn't one of its natively-integrated providers (openai/anthropic/azure/bedrock/gemini/etc.) — added to `requirements.txt`.
  - Groq's free tier (8000 tokens/minute) is tight enough that a ReAct-style agent loop calling multiple tools (get_filing + get_price + get_news) in one task run reliably blows the budget, since each turn resends the full growing conversation history. Fixed by: (1) trimming what each tool returns before it reaches the agent's context (`agents/tools.py` — filing text capped at 2500 chars starting from the actual cover page rather than the XBRL tag-data preamble that precedes it in the stripped HTML, price history capped to the last 5 points, news summaries capped to 200 chars), (2) `max_iter=2` per agent (one tool-call turn, one final-answer turn), (3) each Phase 3 task instructs the agent to call `get_filing` only, skipping `get_price`/`get_news`, and (4) `agents/llm.py`'s `run_with_rate_limit_backoff` retries a `RateLimitError` by sleeping the exact wait time Groq's error message reports. This is a real free-tier constraint, not a one-off bug — worth revisiting once Phase 5's RAG retrieval replaces raw tool dumps with much smaller retrieved chunks, and again in Phase 4 when the debate loop needs multiple sequential agent turns.
  - `get_filing`'s stripped HTML started with a long block of inline-XBRL tag data (fund identifiers, GAAP taxonomy refs) with no prose in it before the actual filing cover page began — for AAPL/MSFT/TSLA the real text started 14k-34k characters in. Initially worked around at the call site (`agents/tools.py` searched for the cover-page marker and truncated from there); **later fixed at the source** in `mcp_server/tools/get_filing.py` — see the "Known issues" entry below for the real fix.

- **Phase 4 — debate loop and Judge agent:**
  - `orchestration/debate_loop.py` holds `speaking_order = [bull_agent, bear_agent]` as the ordered-list pattern spec.md requires; round 1 calls each side's `run_opening_statement` (independent, as in Phase 3), rounds 2+ call `run_rebuttal(company, transcript_text)` which feeds the full transcript-so-far back in and instructs each side to engage the opponent's most recent specific point before adding anything new.
  - `agents/judge_agent.py` is new: reads the finished transcript and scores `stronger_side` / `claims_checked` / `claims_unsupported` / `memo`. Deliberately does its own reading rather than calling any evidence-retrieval tool — real claim-level verification (extraction -> retrieval -> entailment) is Phase 6's shared pipeline, applied uniformly to Bull, Bear, Judge, and deep-dive sub-agents alike; the Judge is not exempt from that later.
  - **New crewai/Groq bug found in Phase 4: a zero-tools agent can still 400.** Giving `run_rebuttal`/the Judge `tools=[]` (the natural way to say "don't call tools") triggers a different code path in crewai's executor (`call_llm_and_parse`, used when `self.original_tools` is empty) that still sends `tool_choice: "none"` to Groq — and `openai/gpt-oss-20b` sometimes emits a tool call anyway (observed calling a hallucinated `browser.get_filing`, and separately a real `get_filing` with a bogus extra arg), which Groq validates strictly and rejects with a 400 (`"Tool choice is none, but model called a tool"`). Worked around by keeping every agent's real tools attached (`tools=ALL_TOOLS`) always, and telling it not to call them via the task description instead — plus `agents/llm.py`'s `run_with_rate_limit_backoff` now also retries this specific `BadRequestError`, since it's model non-determinism (a retry usually doesn't repeat the same errant tool call).
  - **`output_pydantic` costs an extra, unretried LLM call.** The Judge originally used `Task(output_pydantic=JudgeVerdict)` for structured output. crewai converts free text into the pydantic model via a second, separate LLM call (through the `instructor` library) when the model doesn't return native structured output — and that conversion call isn't covered by our `run_with_rate_limit_backoff` wrapper, so it fails hard on any rate limit. Replaced with a plain structured-text format (`STRONGER_SIDE: ...` / `CLAIMS_CHECKED: ...` / etc.) that the Judge is instructed to output and `agents/judge_agent.py`'s `_parse_verdict` parses with a regex — no extra LLM call, no extra failure point.
  - **Groq free tier has a daily cap too, not just per-minute.** `openai/gpt-oss-20b` on this account is capped at both 8000 tokens/minute *and* 200,000 tokens/day. Heavy same-day testing (this Phase 3 + Phase 4 session) ran the daily quota down to ~0 remaining — confirmed via a `RateLimitError` reporting "on tokens per day (TPD): Limit 200000, Used 199965". `run_with_rate_limit_backoff`'s retry-after parser only handled seconds ("Xs"), not the "Xm Y.Zs" format Groq uses for longer (daily-cap) waits, so it was retrying every ~15s against a 20+ minute wait and burning through all 6 attempts uselessly. Fixed: the parser now handles both formats, and any parsed wait over `MAX_SENSIBLE_WAIT_SECONDS` (90s) is treated as the daily cap and fails fast with a clear message instead of blocking or retrying futilely.
  - **Not yet re-verified after the last two fixes** (Judge's plain-text parser, the daily-cap fail-fast): the daily quota ran out before a fresh end-to-end run could confirm them together. The debate loop *was* fully verified end-to-end before those fixes (`scripts/test_phase4_debate.py` on AAPL — 3 rounds, every round-2+ statement named and rebutted a specific opponent claim, Judge produced a verdict), so the core mechanism is proven; re-run `python -m scripts.test_phase4_debate AAPL` once the daily quota resets to confirm the memo no longer truncates.
  - Also bumped `orchestration/debate_loop.py`'s `INTER_TURN_PAUSE_SECONDS` to 15s between agent turns, to spread requests out against the per-minute cap and reduce (not eliminate) how often the retry path is needed at all.

- **Phase 5 — RAG index (Chroma):**
  - `rag/chunker.py`: fixed-window chunking with overlap (1000 chars / 200 overlap, snapped to word boundaries), sized to `all-MiniLM-L6-v2`'s 256-token truncation limit. Not the Item-N section-aware split — that's still Phase 7's `deep_dive/section_splitter.py`, run separately for deep-dive filtering. `chunk_filing()` takes a `get_filing()` data dict and returns records with deterministic ids (`{company}-{filing_type}-{section}-{chunk_index}`, no random component — see the id-collision bug below for why that matters) and metadata (company, filing_type, filed_date, source_url, section, chunk_index).
  - `rag/embedder.py`: local `sentence-transformers` (`all-MiniLM-L6-v2`), no API key, cached as a module-level singleton since loading the model is the slow part.
  - `rag/chroma_store.py`: one persistent Chroma collection per company (`filing_{TICKER}`), not one shared collection with a metadata filter — simple physical isolation, good enough for Phase 5's single-user testing. Phase 10's `orchestration/session.py` adds per-debate-session isolation on top of this later.
  - `rag/reranker.py`: cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks Chroma's vector-similarity candidates — slower but more precise than bi-encoder similarity alone, run only over the small candidate set Chroma already narrowed down to.
  - `rag/retriever.py`: `index_company_filing(company, filing_type)` ties chunker+embedder+chroma_store together; `retrieve(query, company, top_k, candidate_k)` ties embedder+chroma_store+reranker together.
  - **Real bug found during acceptance testing: chunk ids had a random UUID suffix, silently breaking "upsert" idempotency.** `chunk_filing()` originally appended `uuid.uuid4().hex[:8]` to every chunk id "to guarantee uniqueness," which meant every call to `index_company_filing()` generated an entirely new set of ids — so `chroma_store.upsert_chunks()`'s upsert-by-id never actually overwrote a previous index, it just kept *adding* duplicate copies of the same filing's chunks (old text alongside new text) every time a company was re-indexed. This went unnoticed until the `get_filing.py` page-footer fix (below) was tested: the retrieval acceptance script re-indexed AAPL and still surfaced pre-fix chunk text containing the old footer noise, because the collection now held both pre-fix and post-fix chunks mixed together and Chroma had no way to know the old ones were stale. Fixed two ways: (1) chunk ids are now fully deterministic (company + filing_type + section + chunk_index, no random component), so a same-length re-index correctly overwrites via upsert; (2) `chroma_store.delete_filing_chunks()` is called before every `index_company_filing()` upsert regardless, since a *shorter* re-index (fewer chunks than before, e.g. after a text-cleaning fix) would still leave the old tail-end chunks orphaned even with deterministic ids. Local `chroma_data/` was wiped and rebuilt clean after this fix — anyone with a pre-existing local Chroma store from before this fix should do the same (`rm -rf chroma_data`) rather than trust upsert to have cleaned it up on its own.
  - **Second real bug, found in `get_filing.py`, surfaced while debugging the above:** AAPL's 10-K (not MSFT's or TSLA's — filer/template-specific) repeats a page-footer string ("`{Company Name}. | {Year} Form {Type} | {Page}`", e.g. "Apple Inc. | 2025 Form 10-K | 3") on every rendered page — confirmed 58 occurrences in one filing. It's pure page furniture with no content, but because it's near-identical across dozens of chunks, it actively hurt retrieval quality: a query landing near a page boundary would match on the shared footer text as easily as on the chunk's real content, pulling in the wrong chunk. Fixed in `mcp_server/tools/get_filing.py`'s new `_strip_page_footers()`, called after `_strip_html()`, using a regex built from the filing's own already-resolved company name (`re.escape(company_name) + r"\.?\s*\|\s*\d{4}\s*Form\s+[\w-]+\s*\|\s*\d+\b"`) so it only ever matches that filing's real footer, never guesses. Verified zero leftover matches on all 7 previously-tested filers (AAPL/MSFT/TSLA/RIOT/NVAX/BABA/TM) and zero regressions (MSFT/TSLA/etc. had 0 matches before and after, since they don't use this footer style).
  - **Acceptance methodology**: no hand-labeled test set exists yet (`evaluation/test_set.json` is Phase 6's job, for claim-level entailment, not retrieval). `scripts/test_phase5_retrieval.py` instead builds a reproducible smoke test from the filing itself — index it, sample chunks spread across the document, treat a snippet from partway through each as a "question," and check whether reranking correctly re-surfaces that exact source chunk in the top 3 out of the company's *entire* chunk set (258 for AAPL, 407 for MSFT). Had to add a distinctiveness filter (snippet must contain a digit or >= 2 proper-noun-like capitalized words) after discovering that a naive random-span sampler frequently landed on genuinely-duplicated legal boilerplate (10-K risk-factor paragraphs routinely close with the identical sentence "...could materially adversely affect the Company's business, reputation, results of operations, financial condition and stock price." verbatim, dozens of times) — no retriever can or should disambiguate between interchangeable instances of the same boilerplate sentence, so scoring those as failures was testing document repetitiveness, not retrieval quality. Final result: 10/10 on both AAPL and MSFT (acceptance bar is >= 8/10).

## Known issues / open bugs

Bugs and rough edges hit so far, not all fully resolved:

- **Confirmed live hallucination: an agent stated a claim with zero basis in the retrieved tool output — nothing catches this today.** Running `python -m scripts.test_phase3_agents MSFT` (2026-08-30), Bear's opening statement asserted *"a slowdown in cloud revenue growth and a modest decline in operating margin"* as if it were a retrieved fact from Microsoft's 10-K. Checked directly: `get_filing_tool`'s actual output for that run was only the filing's cover page (2500 chars — company info, ticker, notes due dates, filer-status checkboxes); it contains no mention of "cloud," "margin," or "slowdown" anywhere. The model filled the gap from its own pretraining knowledge of Microsoft, not from anything it retrieved, directly violating `BEAR_SYSTEM_PROMPT`'s "only make claims that are backed by retrieved evidence" instruction. That instruction is currently prompt-level only — nothing enforces or checks it. This is precisely the gap **Phase 6** (claim extraction -> retrieval -> entailment pipeline) is designed to close; until that's built, treat every Bull/Bear/Judge claim as unverified regardless of how confidently it's phrased or cited. Worth using this exact example (MSFT Bear, "cloud revenue growth"/"operating margin" claim) as a regression case once Phase 6's entailment checker exists — it should flag it as `not_enough_evidence` or `contradicted`.
- **Bull/Bear/Judge still call `get_filing_tool` and its 2500-char cover-page truncation directly — not yet wired to Phase 5's `rag.retriever.retrieve()`.** This was flagged in Phase 3 as a workaround for Groq's token cap, to be revisited "once Phase 5's RAG retrieval shrinks what a fact costs in tokens" — Phase 5 is now built and verified (10/10 retrieval accuracy), but the agents themselves haven't been switched over to use it yet. Until that wiring happens, Bull/Bear/Judge are still limited to whatever fits in the filing's cover page, and can't see price/news data either. This is the natural next integration step, not a new bug — just noting it's still open.
- ~~`get_filing`'s underlying raw text starts with tens of thousands of characters of inline-XBRL tag noise before real prose begins~~ **Fixed at the source, verified against smaller/foreign filers plus a content diff.** `mcp_server/tools/get_filing.py`'s `_strip_html` now decomposes any element with inline `display:none` styling (verified against real EDGAR HTML: the hidden `<ix:header>` block — fund identifiers, GAAP taxonomy refs, duplicate/hidden facts — is wrapped in `style="display:none"`, while visible inline-tagged values like `ix:nonfraction` sit in normally-styled elements and must not be touched) plus an explicit `ix:header` decompose as a belt-and-suspenders fallback.
  - Verified the fix against 7 filings, not just the 3 large standard filers: AAPL/MSFT/TSLA (large accelerated filers), **RIOT and NVAX (smaller-cap 10-K filers)**, and **BABA and TM (foreign private issuers, 20-F)**. Cover-page text now starts within ~30-80 characters of `raw_text` for all 7 (was 14k-90k chars in before, worse for the bigger foreign filings).
  - **This surfaced two real bugs in the first pass of the fix, caught specifically by testing RIOT's messier HTML** (exactly the kind of filer the smaller/less-standardized hypothesis predicted would break it): (1) `tag["style"]` crashed with `TypeError: 'NoneType' object is not subscriptable` on a bare `style` attribute with no value; (2) decomposing a `display:none` parent sets bs4's `.attrs = None` on the whole subtree, but `soup.find_all(style=True)` is computed once up front, so a nested `display:none` child (RIOT's HTML nests them) still showed up in the loop after its parent was already gone, crashing the same way. Both fixed with a `tag.attrs is None: continue` guard before touching `tag.attrs.get("style")`.
  - **Ran a content diff, not just a "starts near character 0" check**, to make sure the `display:none` sweep isn't eating real hidden-but-legitimate disclosure text (some filers do hide real boilerplate via CSS, not just XBRL tags). For each filing: compared old-strip vs new-strip word sets, captured the actual text of every decomposed element, and ran a prose-heuristic (common-word ratio) over each removed block. Result across all 7 filings: **zero removed blocks flagged as prose-like**; every "lost" word is an unspaced XBRL/GAAP taxonomy element name (e.g. `accumulatedothercomprehensiveincomemember`, `commonstockmember`), never real vocabulary. The heuristic itself was sanity-checked against a real legal-boilerplate sentence (correctly flagged prose-like) and a taxonomy-name string (correctly flagged not).
  - `agents/tools.py`'s `get_filing_tool` no longer needs its marker-search workaround — simplified back to a plain prefix truncation. Phase 5's chunker inherits clean text for free.
- **`crewai` 1.15.18's `cache_breakpoint` bug is a monkeypatch, not a real fix** (`agents/llm.py`, patches `crewai.llms.cache.mark_cache_breakpoint`). It works because both crewai executors re-import that name locally per call, but it's brittle to a crewai internals change — pin crewai's version or re-verify this patch still applies before upgrading crewai later.
- **Windows console can't print LLM output containing smart quotes/en-dashes by default.** `python scripts/test_phase3_agents.py` crashed with `UnicodeEncodeError` on cp1252 when printing a Bull statement containing `’`/`‑` characters. Fixed locally in that script with `sys.stdout.reconfigure(encoding="utf-8")`, but any other CLI script (or the Phase 10 worker's logging) that prints raw LLM output on Windows will need the same fix — not yet applied project-wide.
- **The `mcp` package version is fragile to reinstall order.** Installing `crewai` silently downgraded `mcp` from v2.1.1 to v1.28.1 as a transitive dependency, which broke `mcp_server/server.py` (`MCPServer` renamed/moved to `FastMCP`) until caught and fixed. If `requirements.txt` is ever reinstalled fresh in a different order, or `crewai` is upgraded, re-run `scripts/test_mcp_tools.py` to confirm the MCP server still boots.
- **Groq's free-tier rate limit is a standing constraint, not a one-off bug — confirmed worse in Phase 4.** A full 3-round debate (7 LLM calls: 2 openings + 4 rebuttals + 1 verdict) reliably brushes up against both the 8000 tokens/minute cap and, after enough same-day testing, the 200,000 tokens/day cap. Current mitigations (trimmed tool output, `max_iter=2`, single-tool-call tasks, 15s inter-turn pauses, retry-with-backoff covering `RateLimitError`/`ValueError`/`BadRequestError`) get one full debate through most of the time, but this account's daily quota was fully exhausted by this session's testing — see the Phase 4 design decisions log above. A real production strategy (paid tier, spreading debates across time, or a much cheaper model for rebuttal/judge turns) is needed before this scales past manual testing.
- **Bull/Bear rebuttal and Judge turns still have real tools attached (`tools=ALL_TOOLS`) even though they're instructed not to call them**, as a workaround for the zero-tools `tool_choice: "none"` 400 bug (Phase 4 design decisions log). This is a prompt-level instruction, not an enforced constraint — nothing stops the model from calling a tool anyway on a given turn, which would both cost extra tokens and go against the "read the transcript, don't fetch new evidence" design intent. Watch for this in transcripts (a rebuttal statement introducing a fact that wasn't in any prior round is the tell).
- **Phase 4's Judge memo-truncation fix and the daily-cap fail-fast haven't been re-verified together end-to-end** — the daily quota ran out before a fresh run could confirm them. Re-run `python -m scripts.test_phase4_debate AAPL` once the quota resets (see Phase 4 design decisions log for details).

## Not yet decided / deferred

- The second (general-domain) project — intentionally set aside for a future session
- Whether to eventually randomize debate turn order
- Whether to add an automatic deep-dive trigger (e.g. if too many unsupported claims are flagged) — deferred as unnecessary complexity for v1
