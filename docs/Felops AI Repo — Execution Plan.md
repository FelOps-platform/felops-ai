# Felops AI Repo — Execution Plan

15/10/2026 → \~15/05/2027 · 4 AI engineers (A, B, C, D)

## 1. Assumptions

- After 3.5 Months = university presentation.
- We build mocks for everything outside the AI repo. Mocks follow the §7 contract and swap by config:
  - **Mock backend:** token endpoint, webhook sink, config fixtures.
  - **Mock MCP server:** separate container, local HTTP, 6 actions (Shopify, QuickBooks, etc.).
- Single API process, asyncio, Postgres, Qdrant. No Celery in V1.
- Multiple sub-agents are in scope (a non-goal in the design doc §10): more external sub-agents (`billing`, `sales`), the selector node and cross-sub-agent handoff. Built in P9–P10. The MVP keeps one sub-agent. Update the design doc's non-goals before P9.
- External channels only. Internal sub-agents (HR, IT), the employee portal and the internal audience are out of this plan. The seam stays: `audience` in `AgentRef` is constant `external`.

## 2. Build order

- Build a **walking skeleton first** (P0), then RAG (P1), then tools (P2), then approval, verify and resume (P3), then config, modes and preview (P4). 
- The MVP is P0–P6. After that, the plan is P7–P10 toward full V1.

## 3. Phases to the MVP (P0–P6)

### P0 — Foundations · 15/10 → 25/10

**Goal:** a repo that boots and answers one message through a stub graph. **Exit:** `make up`, then POST a message and get an LLM reply.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | Repo skeleton: `ports/ domain/ adapters/ orchestrator/`, import-linter, CI | §5 | D | 2 | – | CI green |
| 0.2 | Docker compose: API, Postgres, Qdrant, mock MCP | §3 | D | 1 | 0.1 | `make up` healthy |
| 0.3 | Domain types: WorkspaceContext, AgentRef (constant), AgentState, event envelope | §2.3, §6.1, §7.2 | A | 2 | 0.1 | Unit tests pass |
| 0.4 | Protocols: LLMProvider, PromptStore | §5.1 | D | 1 | 0.1 | Contract test stub |
| 0.5 | Protocols: VectorStore, EmbeddingModel | §5.1 | B | 1 | 0.1 | Contract test stub |
| 0.6 | Protocols: Connector (optional idempotency key), MemoryStore | §5.1 | C | 1 | 0.1 | Contract test stub |
| 0.7 | LLM adapter (LiteLLM, one provider) + file PromptStore | §4 Gateway, Prompt Registry | D | 2 | 0.4 | Completion works |
| 0.8 | Mock backend: token endpoint, webhook sink, config fixtures | §7.1 | C | 3 | 0.1 | Endpoints answer |
| 0.9 | Walking skeleton graph: router → responder, in-memory checkpoint | §6.3 | A | 3 | 0.3 | Canned reply via graph |
| 0.10 | Fixtures: AR+EN docs, 20 Q&A, 6 action specs | §8.2 | B | 2 | – | Files in repo |

### P1 — RAG slice · 26/10 → 08/11

**Goal:** ask a question, get a streamed answer grounded in an uploaded source. **Exit:** `demo_p1.sh` runs add source → ask → streamed answer with citation.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Qdrant VectorStore adapter; rejects queries with no workspace filter | §2.3, §5.1 | B | 2 | 0.5 | Filter test passes |
| 1.2 | BGE-M3 EmbeddingModel in thread pool | §3, §4 Embedding | B | 2 | 0.5 | API loop not blocked |
| 1.3 | Ingestion v0: text/TXT → fixed chunks → embed → upsert | §8.5 | B | 3 | 1.1, 1.2 | Chunks in Qdrant |
| 1.4 | Retriever v0: dense top-k with citations | §4 Retriever | B | 2 | 1.3 | Recall on 5 Q&A |
| 1.5 | Graph v1: guardrails stub → router (rules) → retriever → reasoner → responder | §6.3 | A | 4 | 0.9, 1.4 | Grounded answer |
| 1.6 | `customer_support` persona + wizard block (file templates) | §4.2 | A | 1 | 0.7 | Used by reasoner |
| 1.7 | POST `/messages` → 202 + job\_id, asyncio task, in-memory Idempotency-Key | §2.2, §7.1 | D | 3 | 0.9 | Same key, same job |
| 1.8 | SSE `/events`: `message.delta`, `message.final`, in-process bus | §7.2 | D | 3 | 1.7 | Curl streams tokens |
| 1.9 | POST `/knowledge-sources`, GET status | §7.1 | D | 2 | 1.3 | Status ready |
| 1.10 | Mock MCP server (6 actions) + Connector adapter; reads take no idempotency key | §4 MCP Client | C | 4 | 0.6, 0.8 | List and call work |
| 1.11 | `effective_tools(P,S)` pure function + property tests | §4.1 inv. 7 | C | 2 | 0.3 | E ⊆ P, E ⊆ S |
| 1.12 | Action specs: schema, side-effect class (read-only or side-effecting), verifier spec | §4.1 | C | 2 | 0.10 | 6 specs load |

### P2 — Tool slice · 09/11 → 22/11

**Goal:** the agent looks up an order through a tool. **Exit:** "Where is my order #1042?" calls mock Shopify and answers. Deselecting the action blocks it.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 2.1 | Tool Registry v0: P manifest file, `effective_for`, `validate_args` | §4.1 | C | 3 | 1.10, 1.11 | E derived from S |
| 2.2 | Executor v0: call via Connector, `tool.*` events, simple backoff, reads and writes split by side-effect class | §4 Executor | C | 3 | 2.1 | Lookup runs |
| 2.3 | Policy Engine v0: allowlist = effective set | §4 Policy | C | 1 | 2.1 | Out-of-set denied |
| 2.4 | Graph v2: reasoner → tool\_selector → policy\_check → executor → responder | §6.3 | A | 4 | 2.1–2.3 | Order answer |
| 2.5 | PUT `/agents/{id}/config` (store snapshot, recompute E), `config:validate` minimal | §7.1, §4.1 | D | 4 | 2.1 | `config_version` returned |
| 2.6 | Events: `retrieval.completed`, `tool.*` with agent and sub-agent ids | §7.2 | D | 2 | 2.2 | Envelope complete |
| 2.7 | Parsers: PDF, DOCX (Docling) | §8.5 | B | 4 | 1.3 | Both ingest |
| 2.8 | Parser: web URL (trafilatura) | §8.5 | B | 2 | 1.3 | URL ingests |
| 2.9 | Retrieval eval seed: Recall@k, MRR on 20 Q&A | §4 Evaluation | B | 2 | 1.4 | Baseline number saved |

### P3 — Approval, verify, resume · 23/11 → 06/12

**Goal:** the full Sara Ali flow (§8.2): ask, approval, approve, send, verify. **Exit:** scripted scenario passes, including resume after a restart.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 | Postgres checkpointer | §6.5, §3 | A | 2 | 2.4 | State survives restart |
| 3.2 | `approval_gate`: `interrupt()`, params snapshot, `params_hash`, `effect_summary` templates | §6.5, §8.2 | A | 4 | 3.1 | Run pauses |
| 3.3 | Policy v1: `requires_approval` rule | §4 Policy | C | 1 | 2.3 | Returns require\_approval |
| 3.4 | POST `/approvals/{id}/decision` → resume task; check hash, TTL, effective set | §7.1, §6.5 | D | 3 | 3.2 | Resume executes |
| 3.5 | Jobs table, GET `/jobs/{id}`, `waiting_approval` | §2.2 | D | 3 | 1.7 | Status correct |
| 3.6 | Idempotency ledger (Postgres) in executor, side-effecting calls only; reads get no key and run live | §4 Executor | C | 3 | 2.2 | Retry makes no duplicate; reads never keyed |
| 3.7 | Verifier v0: one live read-only check per action spec, no idempotency key | §4 Verifier | C | 3 | 2.1 | Pass and fail cases |
| 3.8 | Planner v0: up to 2 steps over effective tools, step and time caps | §4 Planner | A | 4 | 2.4 | `plan.created` |
| 3.9 | Verify-fail path: single check, no retry, straight to `handoff=verify_failed` | §6.5 | A | 2 | 3.7 | Agent never claims success |
| 3.10 | Webhook sender to mock backend (`job.completed`, `approval.required`) | §7.1 | D | 2 | 3.5 | Sink receives |
| 3.11 | Ingestion status machine, retry, delete, content-hash dedupe | §8.5 | B | 4 | 2.7 | Failed source retries |
| 3.12 | Chunking v1: heading-aware | §8.5 | B | 3 | 3.11 | Headings in metadata |

### P4 — Config, modes, preview · 07/12 → 20/12

**Goal:** the wizard flow works from API calls. **Exit:** configure → preview → activate → chat → approve → human takeover.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 4.1 | Prompt Registry v1: Postgres versions, resolution order, wizard block compile, pin by `config_version` | §4, §2.3 | D | 4 | 2.5 | Rollback works |
| 4.2 | Conversation Manager: modes, human\_control forward-only, `active_sub_agent_id` constant | §4, §6.4 | A | 3 | 3.1 | Agent silent in takeover |
| 4.3 | POST `/conversations/{id}/mode` | §7.1 | A | 1 | 4.2 | Mode switches |
| 4.4 | Events table with `seq`, `Last-Event-ID` replay, GET `/trace` | §2.2, §7.1 | D | 3 | 2.6 | Replay after reconnect |
| 4.5 | `/preview`: write tools mocked, reads live | §7.1 | C | 3 | 2.2 | No side effects |
| 4.6 | Guardrails v1: injection regex, output language and forbidden-content checks | §4 Guardrails | A | 3 | 1.5 | Test set blocked |
| 4.7 | Router v1: small-LLM intent | §6.3 | A | 2 | 4.2 | 90% on test set |
| 4.8 | Memory v0: window + rolling summary, `memory_writer` node | §4 Memory | B | 4 | 1.5 | Multi-turn context kept |
| 4.9 | Arabic normalization (diacritics, alef/yā) | §8.5 | B | 2 | 3.12 | Test passes |
| 4.10 | `config:validate` full: `tool_unreachable`, `effective_tools`, source warnings | §7.1, §4.1 | C | 3 | 2.5 | Report correct |

### P5 — MVP build-out · 21/12 → 10/01 (feature freeze 10/01)

**Goal:** the live demo runs from the console, stable enough to rehearse. **Exit:** all §8.2 steps pass in CI. Tag `rc1`.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 5.1 | Hybrid retrieval: Qdrant sparse BM25 + RRF | §4 Retriever | B | 5 | 4.9 | Beats dense on eval |
| 5.2 | Demo console: chat, trace, approvals, KB upload, action toggles | §7.2, §8 | D | 6 | 3.4, 4.4 | Full flow in browser |
| 5.3 | Handoff paths: low\_confidence, verify\_failed, assistance requested | §6.5 | A | 3 | 4.2 | All three trigger |
| 5.4 | Model Gateway v1: retry, fallback provider, `usage.recorded` | §4 Gateway | C | 4 | 0.7 | Fallback on forced failure |
| 5.5 | Output guardrail loop: regenerate once, fixed fallback | §6.4 | A | 2 | 4.6 | Violation handled |
| 5.6 | Demo dataset: 2 workspaces, AR+EN KB, scripted scenarios | §8.2 | B | 3 | 3.12 | Reset script works |
| 5.7 | E2E scenario tests in CI (§8.2 steps 1–11) | §4 Evaluation | C | 4 | 3.10 | Green nightly |
| 5.8 | Startup recovery: resume `running` jobs, re-run `processing` sources | §2.2 | C | 2 | 3.5 | Kill test passes |
| 5.9 | Cross-workspace leakage test v0 (2 workspaces) | §2.3 | B | 2 | 5.6 | Zero leaks |
| 5.10 | Latency pass: time to first token, caches | §2.2 | A | 3 | 5.5 | First token under 3 s |

### P6 — Stabilization · 11/01 → 28/01 (demo Fri 29/01)

**Goal:** nothing new, everything solid. **Rules:** after 10/01 only fixes tied to a bug id. After 27/01 nothing changes. **Exit:** tag `mvp-1.0`, 3 clean rehearsals.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 6.1 | Bug bash, P0/P1 bugs only | all | All | 5 | 5.\* | Zero open P0/P1 |
| 6.2 | Clean-install test on 3 machines | §3 | D | 2 | 6.1 | `make up` to demo works |
| 6.3 | Failure drills: LLM down, Qdrant restart, approval expiry, tool error | §2.2, §6.5 | C | 3 | 6.1 | Each recovers visibly |
| 6.4 | AR+EN rehearsal, 50-question quality spot check | §4 Retriever | B | 3 | 6.1 | 90% grounded |
| 6.5 | Harden approval, handoff and takeover scenarios | §6.5, §8.3 | A | 3 | 6.1 | 10 runs, no failure |
| 6.6 | Load check: 10 concurrent chats | §2.2 | C | 1 | 6.3 | No errors |
| 6.7 | Lock demo script (10 min) + backup screen recording | §8.2 | D | 2 | 6.2 | Recording saved |
| 6.8 | Dress rehearsals ×3 on the presentation setup and network | all | All | 3 | 6.7 | Timed, no surprises |

## 4. MVP definition

One live product demo: the customer-support agent in the demo console, using the Sara Ali story (§8.2).

| Works live | Mocked | Deferred |
| --- | --- | --- |
| Create agent config (API), add sources (text, TXT, PDF, DOCX, URL), AR+EN | Backend (auth, config store, billing, channels) | Reranker, long-term memory, multi-step replan |
| Hybrid retrieval with citations, streamed answers | MCP server and the 6 actions (Shopify, QuickBooks…) | Presidio, injection classifier, ModelPolicy |
| Order lookup tool, invoice send with approval, verify | UI (our demo console only) | Langfuse, full eval suites |
| Approve and reject, resume after restart | Webhook receiver, token broker | Tier routing, local vLLM |
| Human takeover, handoff on low confidence or verify fail | WhatsApp and widget delivery (SSE only) | Multiple sub-agents, selector node, handoff (P9–P10) |
| Tool selection `S` changes what the agent can do | Real credentials | Dedicated collections, pgvector |
| Event trace and replay |  | All "Future" features |

## 5. Per-module progression map

Simple → upgrades → final. Phase in brackets.

| Module | Simple | Upgrades | Final |
| --- | --- | --- | --- |
| Model Gateway | One provider, direct (P0) | Retry, fallback, usage (P5) | Tier routing, vLLM (P8) → ModelPolicy, budget (P9) |
| Embedding | BGE-M3 local (P1) | – | Alt adapter (P10) |
| Vector store | Qdrant dense, workspace filter (P1) | Sparse vectors (P5) | Dedicated collection (P9), pgvector adapter (P10) |
| Retriever | Dense (P1) | Hybrid + RRF (P5) | Rerank, per-agent strategy (P8) |
| Ingestion | Text/TXT, fixed chunks (P1) | PDF/DOCX/URL (P2) → status, retry, dedupe, heading chunks (P3) → Arabic normalization (P4) | Per-source-type chunking (P8) |
| Prompt Registry | File templates (P0–P1) | Postgres versions, resolution order (P4) | Eval gating, rollback (P8) |
| Planner | Direct tool call (P2) | Up to 2 steps (P3) | Multi-step, deps, replan (P7) |
| Tool Registry | `effective_tools` function (P1) | Manifest + `effective_for` (P2) → recompute + validate (P4) | Zero out-of-set gate in eval (P8) |
| MCP Client | Mock connector (P1) | Token fetch from mock backend (P2) | Pooling, timeouts (P7) |
| Executor | Call + events (P2) | Idempotency ledger, side-effecting calls only (P3) | Backoff, redelivery tests (P7) |
| Verifier | Single live read-only check (P3) | Fail goes straight to handoff (P3) | Evidence templates, spec coverage (P7) |
| Policy Engine | Allowlist (P2) | + `requires_approval` (P3) | + amount threshold, rate limit (P9) |
| Memory | None | Window + summary (P4) | Long-term facts (P7) |
| Conversation Manager | Thread state (P1) | Modes, takeover (P4) | Handoff context attached (P5) |
| Guardrails | Stub (P1) | Regex, language checks (P4) → regenerate loop (P5) | Presidio, classifier, groundedness (P9) |
| Jobs and events | asyncio + in-process bus (P1) | Jobs table (P3) → events table, replay (P4) → recovery (P5) | Backoff retries (P7) |
| Evaluation | Recall@k seed (P2) | E2E scenarios in CI (P5) | Full suites, Ragas, CI gate (P8) |
| Observability | JSON logs (P0) | Trace events (P4) | Langfuse (P8) |
| Isolation | Workspace filter (P1) | Leakage test, 2 workspaces (P5) | CI contract test: retrieval, memory, cache (P9) |
| Sub-agent seams | `AgentRef` constant (P0) | Persona and tool keys per sub-agent (P4) | Registry + billing sub-agent (P9) → selector, third sub-agent, handoff (P10) |
| Cross-sub-agent handoff | Handoff to human (P5) | Thread pinning (P10) | Sub-agent switch with context (P10) |

## 6. Work split


| Phase | A | B | C | D |
| --- | --- | --- | --- | --- |
| P0 | Domain types, skeleton graph | Vector/embedding ports, fixtures | Connector port, mock backend | Repo, CI, compose, LLM adapter |
| P1 | Graph v1, persona | Qdrant, embeddings, ingestion, retriever | Mock MCP, `effective_tools` | Messages API, SSE, sources API |
| P2 | Graph v2 (tools) | PDF/DOCX/URL, eval seed | Tool Registry, executor, policy | Config API, events |
| P3 | Checkpoint, approval gate, planner | Status machine, chunking v1 | Ledger, verifier, policy v1 | Decision API, jobs, webhooks |
| P4 | Modes, guardrails v1, router | Memory v0, Arabic normalization | Preview, validate | Prompt Registry, events table |
| P5 | Handoffs, output loop, latency | Hybrid, demo data, leakage test | Gateway v1, E2E tests, recovery | Demo console |
| P6 | Approval and takeover hardening | AR+EN quality check | Failure drills, load | Clean install, demo script |
| P7 | Runtime reliability | API and platform tests | Planner v1, tuning | Long-term memory |
| P8 | Gateway v2, tool evals | Langfuse, CI eval, prompt gate | τ calibration, dataset | Rerank, chunking, retrieval evals |
| P9 | ModelPolicy, policy chain, billing sub-agent | Leakage check, contract review | Guardrails v2, red team, registry | Isolation CI, dedicated collection, effective-set tests |
| P10 | Third sub-agent, handoff, load test | Docs, OpenAPI, runbook, API fields | Selector node, pinning | Protocol tests, adapters, sub-agent evals |

## 7. Stabilization window

Part of P6, 11/01 → 28/01. Feature freeze Sat 10/01.

| Week | Focus |
| --- | --- |
| 11–15/01 | Bug bash, fixes only. Clean-install on 3 machines. |
| 18–22/01 | Failure drills. AR+EN quality check. Demo script locked. Backup recording. |
| 25–28/01 | 3 dress rehearsals on the real setup. Tag `mvp-1.0` on 27/01. 28/01: no changes. |

The three weeks before the freeze (P5) are the slip buffer for the MVP.

## 8. Months 3.5 → 7: toward full V1

### P7 — Runtime reliability · 31/01 → 20/02

**Exit:** `kill -9` mid-run recovers. A multi-step plan scenario passes.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 7.1 | Retro, triage demo bugs into backlog | all | All | 1 | 6.8 | Backlog ranked |
| 7.2 | Job runner: exponential backoff, `failed` state, `error` event | §2.2 | A | 3 | 5.8 | Forced failure retried |
| 7.3 | Crash-recovery test: kill during run, resume from checkpoint | §2.2 | A | 2 | 7.2 | Test in CI |
| 7.4 | Ledger redelivery tests (side-effecting calls only), webhook retries | §4 Executor | A | 2 | 3.6 | No duplicates; no read call carries a key |
| 7.5 | MCP Client: session pooling, timeouts | §4 MCP Client | A | 3 | 2.2 | Timeout test passes |
| 7.6 | Verifier v1: evidence and reason templates, spec coverage test for every action | §4 Verifier | A | 3 | 3.7 | All 6 actions covered; failure shows evidence |
| 7.7 | Planner v1: multi-step, dependencies, replan on failure | §4 Planner | C | 6 | 3.8 | 3-step plan runs |
| 7.8 | Reasoner and router tuning from demo failures | §6.2 | C | 3 | 7.1 | Backlog items closed |
| 7.9 | Long-term memory facts, workspace-scoped | §4 Memory | D | 5 | 4.8 | Fact recalled next day |
| 7.10 | Idempotency keys to Postgres, `Last-Event-ID` tests | §2.2 | B | 3 | 4.4 | Survives restart |

### P8 — Quality · 21/02 → 20/03

**Exit:** eval report vs the MVP baseline. CI blocks regressions.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 8.1 | Reranker bge-reranker-v2-m3 in thread pool | §3, §4 Retriever | D | 4 | 5.1 | MRR up on eval |
| 8.2 | Retrieval strategy per agent via config | §5.2 | D | 2 | 8.1 | Switch by config |
| 8.3 | Chunking per source type | §8.5 | D | 3 | 3.12 | Eval not worse |
| 8.4 | Retrieval eval suite: Recall@k, MRR, nDCG, Ragas groundedness | §4 Evaluation | D | 4 | 2.9 | Report generated |
| 8.5 | Tool-call and trajectory suites; zero out-of-set calls | §4.1 inv. 7 | A | 5 | 5.7 | Gate green |
| 8.6 | Langfuse tracing | §3 | B | 4 | 4.4 | Traces visible |
| 8.7 | Evals in CI + sampled-trace evals | §4 Evaluation | B | 3 | 8.4, 8.5 | PR blocked on regression |
| 8.8 | Model Gateway v2: tier routing, vLLM adapter | §4 Gateway | A | 6 | 5.4 | Route by tier |
| 8.9 | Prompt versioning with eval gate and rollback | §4 Prompt Registry | B | 4 | 4.1, 8.7 | Bad prompt rejected |
| 8.10 | Confidence threshold τ per agent, calibrate | §6.5 | C | 3 | 8.4 | Handoff rate on target |
| 8.11 | Grow datasets: 100 Q&A (AR+EN), 30 tool scenarios | §4 Evaluation | C | 4 | 2.9 | Datasets in Langfuse |

### P9 — Safety, isolation, sub-agent foundation · 21/03 → 10/04

**Exit:** leakage test shows zero leaks. Red-team set passes target. Two sub-agents load from manifests.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 9.1 | Guardrails v2: Presidio PII, injection classifier, groundedness | §4 Guardrails | C | 5 | 4.6 | Eval pass |
| 9.2 | Red-team set (50 attacks), fix gaps | §4 Guardrails | C | 3 | 9.1 | 95% blocked |
| 9.3 | Isolation contract test in CI: retrieval, memory, cache | §2.3 | D | 4 | 5.9 | Zero leaks |
| 9.4 | Dedicated-collection option (enterprise) | §2.3 | D | 3 | 9.3 | Config switch works |
| 9.5 | ModelPolicy: providers, local-only flag, budget cap | §2.3 | A | 4 | 8.8 | Cap blocks calls |
| 9.6 | Policy chain: amount threshold, rate limit | §4 Policy | A | 3 | 3.3 | Rules tested |
| 9.7 | Output cross-workspace leakage check | §4 Guardrails | B | 2 | 9.3 | Planted leak caught |
| 9.8 | Review AI↔backend contract: token fetch, header trust | §7, §10 | B | 3 | 3.10 | Findings fixed |
| 9.9 | SubAgentRegistry, file manifest loader, `sub_agents` config field, StaticResolver reads the agent's default sub-agent | §4, §5.1, §7.1 | C | 4 | 4.2 | Two manifests load; config picks one |
| 9.10 | `billing` sub-agent (external): manifest P=\[0,1,3,5\], persona templates, mock billing actions | §4.1, §4.2 | A | 4 | 9.9 | Agent set to billing uses its own persona and tools only |
| 9.11 | Effective-set tests with two real sub-agents (§4.1 propagation table) | §4.1 inv. 7 | D | 3 | 9.10 | All table cases pass; support cannot call billing-only tools |

### P10 — Multi-sub-agent and completion · 11/04 → 29/04

**Exit:** all of V1 is built, including multiple sub-agents, a third external sub-agent and cross-sub-agent handoff. Other Future features excluded. If P10 slips, cut 10.4 (handoff) first, then 10.3 (third sub-agent). Selector and billing stay.

| ID | Task | Module (§) | Own | Days | Dep | Done when |
| --- | --- | --- | --- | --- | --- | --- |
| 10.1 | `sub_agent_selector` node on the guardrails(in) → router edge. Audience stays `external`; sub-agent chosen by a small LLM among the agent's enabled sub-agents | §6.6, §5.1 | C | 5 | 9.9, 4.7 | 90% correct on a 30-message routing set |
| 10.2 | Thread pinning: `Thread.active_sub_agent_id`, `agent_ref` pinned in checkpoint, resume skips selection | §6.6, §4 Conversation Manager | C | 2 | 10.1 | Resume keeps the sub-agent |
| 10.3 | Third external sub-agent `sales`: manifest, predefined tool ids, persona templates, mock actions | §4.1, §4.2 | A | 3 | 9.10 | Agent set to sales uses its own persona and tools only |
| 10.4 | Cross-sub-agent handoff: explicit switch mid-thread with summary and plan context, `handoff.requested` reason `sub_agent_switch`, pending approvals revoked | §6.6, §6.5 | A | 4 | 10.2, 10.3 | Support → billing keeps context; no stale approval runs |
| 10.5 | API and events: `sub_agents` list, `effective_tools` map per sub-agent, `sub_agent_id` in every event and usage record, console sub-agent badge | §7.1, §7.2 | B | 3 | 10.1 | Fields visible in console |
| 10.6 | Sub-agent evals: routing accuracy across 3 sub-agents, per-sub-agent tool calls (zero out-of-set), cross-sub-agent leakage | §4 Evaluation, §6.6 | D | 4 | 10.1, 10.3, 8.5 | Gates green in CI; billing-only tools unreachable from support |
| 10.7 | pgvector adapter + alt embedding adapter, swap by config | §5.2 | D | 4 | 1.1 | Swap passes eval |
| 10.8 | Contract test suite per Protocol, all adapters | §5 | D | 3 | 10.7 | Green in CI |
| 10.9 | Load test and tuning, with 3 sub-agents | §2.2 | A | 3 | 10.4 | Target met |
| 10.10 | OpenAPI, runbook, architecture docs, report chapters. Update design doc §10 non-goals | §7, §10 | B | 5 | – | Reviewed |
| 10.11 | Gap review: every doc section vs repo | all | All | 1 | 10.1–10.10 | Checklist closed |

### Buffer · 30/04 → 01/05

Slipped tasks first. Stretch only if P10 exit is met and eval is green. P7 and P8 also hold about 45% free capacity, which is the main slip buffer.

### P11 — Final stabilization · 02/05 → 15/05

**Rules:** freeze Sat 02/05. Then same drill as P6: full E2E and eval regression, failure drills, 3 rehearsals, final tag `v1.0`, report and slides.

### Stretch goals (in order)

1. Voice channel adapter (§9).
2. Internal portals, internal sub-agents (HR, IT) and ACL-filtered employee retrieval (§9).
3. Neo4j `graph` RetrievalStrategy behind the port (§9).

### Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Slip in P1–P3 | Cut list, in order: per-source chunking, LLM router, memory v0, Arabic normalization. Never cut approval flow, tools or console. |
| Mock drifts from real backend | Generate mocks from the §7 OpenAPI. Contract tests. One short sync with backend each week. |
| LLM outage or cost at demo | Fallback provider (P5), recorded video, capped budget, cached demo run. |
| Weak Arabic quality | Eval set from P2, normalization P4, hybrid P5, spot check P6. |
| Slow BGE-M3 or reranker | Thread pool, batching, embedding cache, hosted fallback adapter. |
| Integration debt from late merges | Trunk-based work. E2E in CI from P1, nightly from P5. |
| One member out | Rotation from P7, pair review on every module, short module docs. |
| Presentation date moves | Freeze stays 3 weeks before. Shift P5 and P6 only. |
| Multi-sub-agent work slips in P10 | Registry and billing sub-agent land early in P9. Cut handoff first, then the third sub-agent. Selector and two external sub-agents stay. Start P10 with the routing set already written. |

## 9. Summary

| Phase | Start | End | Key deliverable |
| --- | --- | --- | --- |
| P0 Foundations | 15/10 | 25/10 | Repo boots, stub graph answers |
| P1 RAG slice | 26/10 | 08/11 | Grounded streamed answer |
| P2 Tool slice | 09/11 | 22/11 | Order lookup via tool |
| P3 Approval, verify, resume | 23/11 | 06/12 | Full Sara Ali flow |
| P4 Config, modes, preview | 07/12 | 20/12 | Wizard flow by API, takeover |
| P5 MVP build-out | 21/12 | 10/01 | Demo console, hybrid RAG, freeze |
| P6 Stabilization | 11/01 | 28/01 | `mvp-1.0`, rehearsed demo |
| **Demo** | **29/01** |  | University presentation |
| P7 Runtime reliability | 31/01 | 20/02 | Crash recovery, multi-step planner |
| P8 Quality | 21/02 | 20/03 | Rerank, evals in CI, Langfuse |
| P9 Safety, isolation, sub-agent foundation | 21/03 | 10/04 | Guardrails v2, ModelPolicy, leakage CI, billing sub-agent |
| P10 Multi-sub-agent, completion | 11/04 | 29/04 | Full V1: 3 sub-agents, selector, handoff |
| Buffer / stretch | 30/04 | 01/05 | Slip and stretch goals |
| P11 Final stabilization | 02/05 | 15/05 | `v1.0`, final presentation |

## 10. Self-check

Each advanced step has its basic step in an earlier phase.

| Advanced (phase) | Needs basic (phase) | OK |
| --- | --- | --- |
| Hybrid retrieval (P5) | Dense retriever (P1) | ✔ |
| Rerank (P8) | Hybrid (P5) | ✔ |
| Planner v1 (P7) | Planner v0 (P3) | ✔ |
| Verifier v1 (P7) | Verifier v0 (P3) | ✔ |
| Executor backoff, redelivery (P7) | Ledger (P3), executor v0 (P2) | ✔ |
| Gateway v2 (P8) | v1 (P5), v0 (P0) | ✔ |
| ModelPolicy (P9) | Gateway v2 (P8) | ✔ |
| Guardrails v2 (P9) | v1 (P4), stub (P1) | ✔ |
| Long-term memory (P7) | Memory v0 (P4) | ✔ |
| Prompt eval gate (P8) | Registry v1 (P4), file (P0) | ✔ |
| Langfuse (P8) | Trace events (P4) | ✔ |
| Full evals (P8) | Seed (P2), E2E (P5) | ✔ |
| Isolation CI (P9) | Leakage test (P5), filter (P1) | ✔ |
| Dedicated collection (P9) | Shared collection (P1) | ✔ |
| pgvector adapter (P10) | Qdrant adapter (P1) | ✔ |
| Sub-agent registry, billing sub-agent (P9) | `AgentRef` constant (P0), persona/tool keys (P4) | ✔ |
| Job backoff, recovery tests (P7) | Jobs table (P3), startup recovery (P5) | ✔ |
| Selector node, thread pinning (P10) | Registry (P9), router v1 (P4) | ✔ |
| Third sub-agent (P10) | Billing sub-agent (P9), mock tools (P1) | ✔ |
| Cross-sub-agent handoff (P10) | Selector, pinning (P10), handoff to human (P5) | ✔ |

Result: no violations.
