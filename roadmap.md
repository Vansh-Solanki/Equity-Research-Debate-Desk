# Equity Research Debate Desk — roadmap

**Purpose:** phase-by-phase build plan. Nothing below has been built yet — this is the execution order once building starts.

## Phase 1 — Free LLM setup
- Create Groq API account, generate free-tier API key
- Test a basic completion call
- Confirm rate limits so later phases (especially the debate loop and deep-dive spawner) respect them
- Status: **Done** — `scripts/test_groq_setup.py` confirms a working completion and prints rate-limit headers. Note: `llama-3.1-8b-instant` is no longer available on this account; using `openai/gpt-oss-20b` instead (1000 req/day, 8000 tokens/min on free tier).

## Phase 2 — Data-tool server (MCP)
- Build a small Python MCP server exposing three tools: `get_filing`, `get_price`, `get_news`
- Wire each tool to its real source: SEC EDGAR API, yfinance, free RSS feed
- Test each tool independently before connecting agents to them
- Status: **Done** — `scripts/test_mcp_tools.py` passes all three tools for AAPL/MSFT/TSLA plus invalid-company error handling. `mcp_server/server.py` registers all three tools on a real MCP server (`mcp` v2, `MCPServer`/`.tool()`). News source is Google News RSS (no key needed, spec's suggested Yahoo RSS feed is deprecated).

## Phase 3 — Bull and Bear agents
- Define both roles in CrewAI
- Give each agent access to the MCP tools
- Confirm each agent runs independent retrieval and writes an opening statement without seeing the other's
- Status: **Done** — `scripts/test_phase3_agents.py` runs Bull and Bear as two separate, non-shared `Crew` kickoffs; both produce opening statements citing >= 2 specific retrieved filing facts (manually spot-checked for AAPL and MSFT). Required patching around several crewai 1.15.18 + Groq free-tier issues — see progress.md.

## Phase 4 — Debate loop and Judge agent
- Implement `speaking_order = [bull_agent, bear_agent]` and loop over it (fixed order for now, one-line change to randomize later)
- Build 2–3 debate rounds where each agent reads the transcript and responds to the other's specific points
- Build the Judge agent to read the full transcript and produce a scored verdict
- Status: **Done** — `orchestration/debate_loop.py` runs `speaking_order` through round 1 (independent openings) then rounds 2-3 (each side rebuts reading the full transcript), then `agents/judge_agent.py` scores it. `scripts/test_phase4_debate.py` produced a full 3-round AAPL transcript where every round-2+ statement explicitly names and rebuts the opponent's specific prior claim (e.g. "Bear's concern about... hinges on the 1.625% notes due 2025 and 2026. In reality...") — spot-checked, acceptance criterion met. Required several more crewai/Groq workarounds — see progress.md.

## Phase 5 — RAG index (Chroma)
- Chunk filings with overlap
- Generate embeddings locally via sentence-transformers
- Store in ChromaDB with metadata (company, filing section, date) for later filtering
- Add cross-encoder reranking on top of retrieval results
- Status: **Done** — `rag/` (chunker, embedder, chroma_store, reranker, retriever) chunks a filing with overlap (1000 chars / 200 overlap, word-boundary snapped), embeds locally via `all-MiniLM-L6-v2`, stores per-company in a persistent Chroma collection, and reranks candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`. `scripts/test_phase5_retrieval.py` scored 10/10 on both AAPL and MSFT (needs >= 8/10) — see progress.md for two real bugs this surfaced (a stale-duplicate-chunk indexing bug and a page-footer noise issue) and how they were fixed.

## Phase 6 — Claim-level evaluation pipeline (hallucination check)
- Build claim extraction (LLM call breaking any agent's output into individual factual claims)
- Run each claim through Chroma retrieval + reranking
- Add local NLI entailment model to classify each claim as supported / contradicted / not enough evidence
- Apply this pipeline uniformly to Bull, Bear, the Judge, and deep-dive sub-agents
- Build the hand-labeled test set (25–30 claims) and measure precision/recall/F1 of the pipeline itself
- Status: **Not started**

## Phase 7 — Deep-dive spawner
- Split filings into sections using their standard labeled structure (Item 1, Item 1A, etc.)
- Generate one button per section dynamically in the UI
- On click, spawn one sub-agent scoped to that section (retrieval filtered by metadata)
- Cache results per section so repeat clicks don't rerun the work
- Merge sub-agent findings back into the main debate context
- Status: **Not started**

## Phase 8 — Evaluation metrics dashboard
- Groundedness % per agent (Bull, Bear, Judge, sub-agents)
- Judge accuracy (F1 against hand-labeled test set)
- Debate engagement score (LLM-as-judge rubric on whether rounds actually respond to prior points)
- Retrieval recall@k (before vs after reranking)
- Latency / LLM calls per debate (and how deep-dive scales it)
- Deep-dive lift (groundedness before vs after a targeted deep dive)
- Status: **Not started**

## Phase 9 — Frontend
- Build the Streamlit UI: ticker input, live debate feed (Bull/Bear message bubbles), deep-dive buttons per section, final memo card with verdict and fact-check count
- Status: **Not started**

## Phase 10 — Deployment
- Wrap frontend + backend + worker into a Docker container
- Build FastAPI backend (async) to receive requests
- Set up Redis job queue (Upstash free tier) to decouple requests from long-running debate jobs
- Build worker pool that pulls jobs, runs the agent crew, isolates each session's Chroma collection
- Deploy container to Hugging Face Spaces (free)
- Set up GitHub Actions for automatic build/redeploy on push
- Status: **Not started**

## Stretch goals (not required for v1, revisit after core build works)
- Randomize debate turn order (trivial change given the ordered-list design)
- Automatic deep-dive trigger if too many unsupported claims are flagged in the main debate
- "Deep dive on everything" mode in addition to per-section buttons

## Explicitly out of scope
- Merger and acquisition analysis (single-company research only)
- The second (general-domain) portfolio project — covers A2A and Claude Agent SDK, to be scoped in a future session
