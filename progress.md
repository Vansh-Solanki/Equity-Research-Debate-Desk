# Equity Research Debate Desk — progress

**Status:** Phase 2 complete (MCP tool server: get_filing, get_price, get_news), Phase 3 pending
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

- **MCP SDK version:** The `mcp` package installed is v2, which renamed `FastMCP` to `MCPServer` (`from mcp.server.mcpserver import MCPServer`) versus the v1 API the spec assumed. Decorator usage (`@mcp.tool()`) is unchanged.
- **News source:** Spec suggested free RSS feeds generically; implemented against Google News RSS (`news.google.com/rss/search`) since it needs no API key and works for arbitrary company names, not just tickers.
- **SEC EDGAR user agent:** Set to `Equity Research Debate Desk vsolanki4565@gmail.com` in `.env` — SEC blocks requests with generic/placeholder user agents.
- **Filing text extraction:** Added `beautifulsoup4` (not in original spec's requirements.txt draft) to strip HTML from 10-K documents into plain text for later chunking in Phase 5. Raw text capped at 3M chars as a safety limit.

## Not yet decided / deferred

- The second (general-domain) project — intentionally set aside for a future session
- Whether to eventually randomize debate turn order
- Whether to add an automatic deep-dive trigger (e.g. if too many unsupported claims are flagged) — deferred as unnecessary complexity for v1
