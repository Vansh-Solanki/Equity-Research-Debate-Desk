# Equity Research Debate Desk — progress

**Status:** Phase 3 complete (Bull and Bear CrewAI agents produce independent, evidence-backed opening statements), Phase 4 pending
**Last updated:** August 29, 2026

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
  - `get_filing`'s stripped HTML starts with a long block of inline-XBRL tag data (fund identifiers, GAAP taxonomy refs) with no prose in it before the actual filing cover page begins — for AAPL/MSFT/TSLA the real text starts 14k-34k characters in. `agents/tools.py`'s `get_filing_tool` now searches for the `"SECURITIES AND EXCHANGE COMMISSION"` cover-page marker and truncates from there instead of from character 0.

## Known issues / open bugs

Bugs and rough edges hit so far, not all fully resolved:

- **Bull/Bear only call `get_filing` right now, not `get_price` or `get_news`.** This is a deliberate Phase 3 workaround for Groq's free-tier 8000 tokens/minute cap (see design decisions log above), not the intended end state — spec section 6 and roadmap Phase 3 both expect agents to have real access to all three tools. Revisit once Phase 5's RAG retrieval shrinks what a "fact" costs in tokens; until then, opening statements are only as good as what's in the truncated filing text.
- **`get_filing`'s underlying raw text (`mcp_server/tools/get_filing.py`) still starts with tens of thousands of characters of inline-XBRL tag noise before real prose begins** (14k-34k chars in, varies by filer). The Phase 3 agent-facing wrapper (`agents/tools.py`) works around this by searching for the `"SECURITIES AND EXCHANGE COMMISSION"` cover-page marker and truncating from there, but the fix is local to that one call site. Phase 5's chunker will hit the same noise when it chunks the full raw filing and needs its own handling (e.g. strip/skip the XBRL block during chunking, not just at agent-tool truncation time).
- **`crewai` 1.15.18's `cache_breakpoint` bug is a monkeypatch, not a real fix** (`agents/llm.py`, patches `crewai.llms.cache.mark_cache_breakpoint`). It works because both crewai executors re-import that name locally per call, but it's brittle to a crewai internals change — pin crewai's version or re-verify this patch still applies before upgrading crewai later.
- **Windows console can't print LLM output containing smart quotes/en-dashes by default.** `python scripts/test_phase3_agents.py` crashed with `UnicodeEncodeError` on cp1252 when printing a Bull statement containing `’`/`‑` characters. Fixed locally in that script with `sys.stdout.reconfigure(encoding="utf-8")`, but any other CLI script (or the Phase 10 worker's logging) that prints raw LLM output on Windows will need the same fix — not yet applied project-wide.
- **The `mcp` package version is fragile to reinstall order.** Installing `crewai` silently downgraded `mcp` from v2.1.1 to v1.28.1 as a transitive dependency, which broke `mcp_server/server.py` (`MCPServer` renamed/moved to `FastMCP`) until caught and fixed. If `requirements.txt` is ever reinstalled fresh in a different order, or `crewai` is upgraded, re-run `scripts/test_mcp_tools.py` to confirm the MCP server still boots.
- **Groq's free-tier rate limit is a standing constraint, not a one-off bug.** Even with all the Phase 3 mitigations (trimmed tool output, `max_iter=2`, single-tool-call tasks, retry-with-backoff), a single Bull or Bear opening statement can still trip a transient `RateLimitError` that needs one retry (observed once in testing, recovered automatically). This will get worse in Phase 4 (multi-round debate = many more sequential LLM calls) and needs a real strategy (e.g. spacing out calls, shorter prompts, or accepting slower debate runs) rather than being solved by truncation alone.

## Not yet decided / deferred

- The second (general-domain) project — intentionally set aside for a future session
- Whether to eventually randomize debate turn order
- Whether to add an automatic deep-dive trigger (e.g. if too many unsupported claims are flagged) — deferred as unnecessary complexity for v1
