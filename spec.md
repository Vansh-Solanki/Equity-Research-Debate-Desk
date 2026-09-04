# Equity Research Debate Desk — spec.md

**Purpose:** implementation-level detail to hand to Claude Code alongside `progress.md` (decisions/context) and `roadmap.md` (phase order). This file defines structure, contracts, and schemas so each phase can be built without guessing.

---

## 1. Project structure

```
equity-debate-desk/
├── agents/
│   ├── bull_agent.py
│   ├── bear_agent.py
│   ├── judge_agent.py
│   └── prompts.py            # all system prompts live here, not inline
├── mcp_server/
│   ├── server.py
│   ├── tools/
│   │   ├── get_filing.py
│   │   ├── get_price.py
│   │   └── get_news.py
├── rag/
│   ├── chunker.py
│   ├── embedder.py
│   ├── chroma_store.py
│   ├── reranker.py
│   └── retriever.py          # combines the above into one retrieve(query, filters) call
├── evaluation/
│   ├── claim_extractor.py
│   ├── entailment_checker.py
│   ├── pipeline.py            # check_statement/check_claim: extraction -> retrieval -> entailment
│   ├── metrics.py             # groundedness %, precision/recall/F1 scoring
│   ├── retrieval_metrics.py   # recall@k, pre- vs post-rerank (Phase 8)
│   ├── engagement_scorer.py   # LLM-as-judge debate engagement rubric (Phase 8)
│   ├── dashboard.py           # Phase 8: assembles all six dashboard metrics
│   └── test_set.json          # hand-labeled 25-30 claims for Judge F1 validation
├── deep_dive/
│   ├── section_splitter.py
│   └── spawner.py
├── orchestration/
│   ├── debate_loop.py        # speaking_order list + round logic
│   └── session.py            # per-session state, isolated Chroma collection handling
├── api/
│   ├── main.py                # FastAPI app
│   ├── routes.py
│   └── schemas.py             # pydantic request/response models
├── worker/
│   ├── worker.py               # pulls jobs from Redis, runs debate_loop
│   └── queue_client.py
├── frontend/
│   └── app.py                  # Streamlit UI — talks directly to orchestration/deep_dive in-process (Phase 9)
├── logging/
│   └── langfuse_client.py
├── tests/
│   └── ...                     # one test file per module above
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## 2. Dependencies (`requirements.txt`)

```
crewai
groq
chromadb
sentence-transformers
fastapi
uvicorn
redis
streamlit
requests
feedparser        # RSS parsing for news
yfinance
python-dotenv
langfuse           # optional, only if using hosted free tier instead of local logging
pydantic
```

Pin exact versions once Phase 1 confirms what's compatible together, don't pin blindly now.

---

## 3. Environment variables (`.env.example`)

```
GROQ_API_KEY=
CHROMA_PERSIST_DIR=./chroma_data
REDIS_URL=redis://localhost:6379
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
SEC_EDGAR_USER_AGENT=your-name your-email@example.com   # EDGAR requires a descriptive user agent string
```

---

## 4. MCP tool contracts

All tools return a consistent shape: `{"success": bool, "data": ..., "error": str | None}`.

**`get_filing(company: str, filing_type: str = "10-K") -> dict`**
- Input: company name or ticker, optional filing type
- Output `data`: `{"company": str, "filing_type": str, "filed_date": str, "raw_text": str, "source_url": str}`
- Error cases: company not found, no recent filing of that type

**`get_price(company: str, period: str = "1y") -> dict`**
- Input: company ticker, optional lookback period
- Output `data`: `{"ticker": str, "prices": [{"date": str, "close": float}], "summary_stats": {"pct_change": float, "avg_volume": float}}`

**`get_news(company: str, limit: int = 5) -> dict`**
- Input: company name, max articles
- Output `data`: `{"articles": [{"title": str, "published": str, "source": str, "summary": str}]}`

---

## 5. Data schemas

**Claim** (used throughout evaluation)
```json
{
  "claim_id": "uuid",
  "agent": "bull | bear | judge | deep_dive_sub_agent",
  "text": "string",
  "source_section": "string or null",
  "retrieved_chunk": "string or null",
  "entailment_label": "supported | contradicted | not_enough_evidence",
  "confidence": "float 0-1"
}
```

**Debate transcript entry**
```json
{
  "round": "int",
  "agent": "bull | bear",
  "statement": "string",
  "claims": ["claim_id", "..."],
  "timestamp": "iso8601"
}
```

**Job** (queue payload)
```json
{
  "job_id": "uuid",
  "company": "string",
  "status": "queued | running | done | failed",
  "created_at": "iso8601",
  "result": "DebateResult or null"
}
```

**DebateResult**
```json
{
  "transcript": ["DebateTranscriptEntry", "..."],
  "judge_verdict": {
    "stronger_side": "bull | bear | tie",
    "memo": "string",
    "claims_checked": "int",
    "claims_unsupported": "int"
  },
  "deep_dive_available_sections": ["string", "..."],
  "deep_dive_results": {"section_name": "DeepDiveResult"}
}
```

---

## 6. Agent prompts (starting point — refine during Phase 3)

**As of the post-Phase-7 RAG-wiring fix (see progress.md): Bull/Bear opening statements no longer call a tool themselves — retrieval is pre-fetched and injected into the task description as text (same pattern as `deep_dive/spawner.py`'s sub-agents), so the system prompts below were updated to match.**

**Bull agent system prompt**
```
You are an equity research analyst arguing the optimistic case for {company}.
You will be given retrieved excerpts from the company's actual filing as evidence.
Only make claims that are backed by that retrieved evidence — never state a fact you were not given.
Build your strongest case for why this is a good investment.
When responding to Bear's argument, directly address Bear's specific points before adding new ones.
```

**Bear agent system prompt**
```
You are an equity research analyst arguing the cautious case for {company}.
You will be given retrieved excerpts from the company's actual filing as evidence.
Only make claims that are backed by that retrieved evidence — never state a fact you were not given.
Build your strongest case for why caution is warranted.
When responding to Bull's argument, directly address Bull's specific points before adding new ones.
```

**Judge agent system prompt**
```
You are an impartial judge reviewing a debate between a Bull and a Bear analyst about {company}.
For every claim either side made, verify it against retrieved evidence from the filing.
Flag any claim that is not supported by evidence.
Score which side made the stronger, better-supported case.
Write a concise final memo summarizing the strongest points from both sides and your verdict.
```

Store these in `agents/prompts.py` as string templates, not hardcoded inline in each agent file, so they're easy to iterate on.

---

## 7. Section-splitting approach (deep-dive spawner)

10-K filings follow a standard `Item N` structure. Approach:
1. Use regex to find standard item headers: pattern like `Item\s+\d+[A-Z]?\.\s+[A-Z]`
2. Common sections to expect: Item 1 (Business), Item 1A (Risk Factors), Item 3 (Legal Proceedings), Item 7 (MD&A / financial discussion), Item 8 (Financial Statements)
3. Split the raw filing text at each matched header, the text between one header and the next becomes that section's content
4. **Fallback for non-standard filings:** if fewer than 3 standard headers are found, fall back to a simple paragraph-count-based chunking (e.g. split into 4 roughly equal parts) and label them generically ("Part 1," "Part 2," etc.) rather than failing
5. Each section's text gets embedded and stored in Chroma with metadata `{"company": ..., "section": ...}` so deep-dive retrieval can filter by section

---

## 8. Acceptance criteria per phase

Use these as the definition of "done" for each roadmap phase:

- **Phase 1:** A test script successfully gets a completion from Groq using the free-tier key
- **Phase 2:** Each of the three MCP tools returns valid data for at least 3 test companies, with error handling verified for an invalid company
- **Phase 3:** Bull and Bear each produce an opening statement for a test company, backed by at least 2 retrieved facts each, without seeing each other's output
- **Phase 4:** A full 3-round debate transcript is produced where round 2+ statements demonstrably reference the opponent's prior specific claim (manually spot-check 3 examples)
- **Phase 5:** Retrieval for a known test question returns the correct chunk in the top 3 results after reranking, for at least 8/10 test questions
- **Phase 6:** The entailment pipeline achieves at least 0.75 F1 on the hand-labeled test set (adjust threshold once real numbers come in)
- **Phase 7:** Clicking a deep-dive button for a specific section returns a section-scoped summary within a reasonable time, and does not reprocess if clicked twice
- **Phase 8:** All six metrics from the roadmap's evaluation dashboard are computed and displayed for at least one full debate run
- **Phase 9:** The Streamlit UI matches the agreed mockup (ticker input, debate feed, deep-dive buttons, memo card) and updates live as a debate runs
- **Phase 10:** The app is reachable via a public Hugging Face Spaces URL, and two simultaneous browser sessions can run different companies without interfering with each other

---

## 9. Known constraints to respect during build

- No paid services anywhere, if Claude Code suggests a tool, confirm it has a genuinely free tier before using it
- Groq free-tier rate limits must be respected, add basic retry/backoff handling in the worker rather than letting calls fail hard
- Debate turn order must be implemented as the ordered list pattern (`speaking_order = [bull_agent, bear_agent]`), not hardcoded agent-by-name calls, so it stays a one-line change to randomize later
- Every claim, statement, and log entry must be labeled by agent name, never by position ("speaker 1")
- This project does not cover merger and acquisition analysis, scope stays single-company research
