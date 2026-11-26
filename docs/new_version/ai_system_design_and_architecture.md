# Felops — AI System Design & Architecture

## 1. Executive note

- **V1 scope: each agent contains exactly one sub-agent, customer-support** (external, customer-facing, action-capable via tools). V1 ships no multi-sub-agent orchestration, no internal employee sub-agents and no sub-agent selection logic. Both are a later refactor, not a redesign. V1 ships only the seams: `AgentRef` in state and events, `SubAgentRegistry`, `SubAgentResolver`, a per-sub-agent persona (Prompt Registry), a per-sub-agent effective tool set (Tool Registry), and a documented graph insertion point (§4.1, §4.2, §6.6).
- **AI repo owns:** orchestration (LangGraph), retrieval and ingestion, model access, policy, guardrails, evaluation. **Backend repo owns:** auth, workspaces, config persistence, billing, channel delivery, connectors, connections and credentials, OAuth, tool verification, MCP servers and the system catalog.
- **Interface:** REST for config and control (one dependency reads the workspace context from the `/workspaces/{workspace_id}` path parameter), async jobs with webhooks for plan execution, and SSE for tokens and trace events. The backend stores every agent config and sends it with each request. The AI repo holds only derived state: the system catalog copy, indexes, effective tool sets, checkpoints (with the run's copy of `agent_config`), traces, events, jobs and the idempotency ledger.

**Terminology (used consistently in this document)**

| Term | Meaning | Id | V1 |
|---|---|---|---|
| **Agent** | The full agent the user creates in the wizard (e.g. "Customer Experience"). Holds the user-written instructions, knowledge sources, channels and the tool selection `S`. A container of sub-agents. | `agent_id` | Many agents per workspace |
| **Sub-agent** | A specialized, platform-defined unit inside an agent (customer_support, later billing, sales, HR, IT). Has its own persona prompt, its own predefined tool set and its own derived effective tool set. | `sub_agent_id` | Exactly one per agent: `customer_support` |
| **System catalog** | The global list of tools the platform offers, across all connectors. Backend-owned, pushed to the AI repo, read-only there (§4.3). | `tool_id` | One catalog |

Rules: the backend always supplies `agent_id` (from the channel binding). The AI repo resolves `sub_agent_id`. Knowledge, instructions, channels and `S` are agent-level. Persona and effective tools are sub-agent-level.

Tool and action are the same item: the contract says **tool** (`tool_id`), the UI says **action**.

---

## 2. System architecture (AI repo)

### 2.1 Layered view

```mermaid
flowchart TB
  CH["Channels: WhatsApp · Chat widget · Hosted page · Custom UI"] --> GW["Backend API Gateway<br/>auth · workspace resolve · rate limit · channel adapters"]
  GW -->|"workspace header: workspace_id, agent_id"| API["AI Service API (FastAPI)"]
  API --> RUN["Run manager<br/>(asyncio background tasks)"]
  RUN --> ORCH["Orchestrator (LangGraph)"]

  subgraph CORE["AI repo: reasoning core"]
    PL["Planner"]
    RT["Retriever"]
    PO["Policy Engine"]
    EX["Executor"]
  end
  ORCH --> PL
  ORCH --> RT
  ORCH --> PO
  ORCH --> EX

  ORCH --> MG["Model Gateway"] --> LLM["openai | anthropic | gemini | vLLM local"]
  RT --> VS[("Qdrant: dense + sparse")]
  ORCH --> PG[("Postgres: checkpoints, traces, events, jobs, idempotency ledger")]

  EX --> TR["Tool Registry"] --> MCP["MCP Client"]
  MCP --> CONN["Connectors / MCP servers (backend repo)"]
  CONN --> EXT["Shopify · QuickBooks · Salesforce · Gmail"]
  EX --> RESP["Responder"]
  RESP --> EVT["Events: SSE stream + webhook to backend"]
  EVT --> GW
```

### 2.2 Async pipeline

```mermaid
flowchart LR
  A["POST /workspaces/{workspace_id}/conversations/{id}/messages<br/>Idempotency-Key"] --> B{"key seen?"}
  B -->|yes| R["return stored 202 + job_id"]
  B -->|no| C["persist job row, start background task"]
  C --> D["202 + job_id"]
  C --> W["Background task: run graph, thread_id = workspace:conversation"]
  W -->|"events (seq)"| S[("Postgres events table + in-process bus")]
  S --> SSE["SSE /workspaces/{workspace_id}/conversations/{id}/events"]
  W -->|"interrupt()"| P["status = waiting_approval, task ends"]
  P -->|"POST /workspaces/{workspace_id}/approvals/{id}/decision"| W2["new background task: same thread_id, loads checkpoint"]
  W -->|"job.completed"| H["webhook to backend"]
```

| Job type | Purpose | Notes |
|---|---|---|
| `interactive` | Live chat turns | Starts immediately as an asyncio task |
| `resume` | Post-approval resumes | Same `thread_id`, loads checkpoint |
| `ingest` | Parse, chunk, embed | Background task with status in Postgres. Sources left in `processing` after a restart are re-run on startup. |
| failed jobs | Transient errors | Retried with exponential backoff inside the task. After the retry limit the job becomes `failed` and an `error` event is emitted. |

- Checkpoints are written after every node. After a crash or restart, jobs left `running` are resumed from the last checkpoint on startup. The job row and checkpoint keep the run's request, including its `agent_config`, so recovery needs no backend call.
- Side effects are protected by the executor's idempotency ledger, not by task semantics.
- Every event except `message.delta` is stored in the Postgres `events` table with its `seq`. The table backs `/trace` and `Last-Event-ID` replay. Live delivery uses an in-process bus, so the dev phase runs a single API process.

### 2.3 Multi-workspace isolation (AI layer)

| Concern | Mechanism |
|---|---|
| `workspace_id` propagation | Set once at ingress by the `get_workspace_context()` dependency (`/workspaces/{workspace_id}/` path parameter) into a `WorkspaceContext` contextvar. Copied into graph state, the background-task context, MCP calls, log and trace baggage. Resume runs read it from their own request, and startup recovery rebuilds it from the job row. Repositories require it as a non-optional argument. |
| Knowledge namespaces | `NameSpace(workspace_id, agent_id, source_id)`. Knowledge is agent-level and shared by all of the agent's sub-agents. A source belongs to exactly one agent. Default: shared collection per embedding model with indexed `workspace_id` payload. Workspace scoping is derived exclusively from `NameSpace.workspace_id`; `workspace_id` is not a filter key and is never duplicated into `Filter`. Enterprise tier: dedicated collection. |
| Model config | `ModelPolicy` per workspace: allowed providers, tier-to-model map, local-only flag for sensitive data, monthly budget cap. The UI's "Model: Automatic (by plan)" is the default tier map. |
| Prompts | Resolution order: workspace+agent+sub-agent override > workspace+sub-agent > platform sub-agent default > platform default. Template versions are pinned per run. |
| Agent scope | `AgentRef(agent_id, sub_agent_id, audience)` is resolved at ingress and pinned in graph state, checkpoint, events and usage records. V1: `sub_agent_id` is always `customer_support`, audience `external`. |
| State | Checkpoint `thread_id = {workspace_id}:{conversation_id}`. Cache keys are prefixed by workspace. |
| Connections and credentials | The AI repo never stores or sees connector secrets and is not told when a workspace connects or disconnects a connector. MCP Client sends `(workspace_id, connector_id, tool_id, args)` and the backend side resolves the connection. V1: one connection per connector per workspace. |
| Verification | Cross-workspace contract test runs in CI: two workspaces, identical queries, assert zero leakage in retrieval, memory and cache. |

---

## 3. Technology stack

| Component | Choice | Role | Why |
|---|---|---|---|
| Orchestration | LangGraph + `langgraph-checkpoint-postgres` | Stateful graph, durable `interrupt()` | Native pause and resume for approvals, checkpoint replay |
| Service | FastAPI + Pydantic v2 | AI Service API, typed contracts | Same stack as backend, shared OpenAPI |
| LLM gateway | In-house `ModelGateway`, LiteLLM as provider adapter, vLLM for local | Routing, fallback, metering | One adapter covers most providers. Routing policy stays ours. |
| Vector DB | Qdrant | Dense and sparse vectors, payload filters | Hybrid in one engine, workspace-indexed filters, self-hostable |
| Lexical (BM25) | Qdrant sparse vectors, IDF modifier | Keyword leg of hybrid | Avoids a separate search cluster |
| Embedding | BGE-M3 (self-hosted); OpenAI-compatible adapter as alternate | Multilingual dense vectors | Strong Arabic and English, 8k context. Inference runs in a thread pool so it never blocks the API event loop. |
| Reranker | bge-reranker-v2-m3; Cohere Rerank adapter | Cross-encoder rerank | Same multilingual coverage, local option. Runs in the same thread pool. |
| Parsing | Docling or Unstructured (PDF/DOCX/TXT), trafilatura (web) | Ingestion extraction | Layout-aware, Arabic-safe |
| Job runner | asyncio background tasks, Postgres `jobs` table | Async plans and ingestion | No broker needed at dev scale. Job state survives restarts. |
| Events | Postgres `events` table + in-process bus | Ordered per-conversation events, SSE and replay | One less service. `seq` gives ordering and `Last-Event-ID` replay. |
| State store | PostgreSQL | Checkpoints, traces, events, jobs, idempotency ledger, catalog copy | One transactional store for graph state and ledger |
| Observability | Langfuse, structured JSON logs | Traces, LLM spans, cost, error logs | Langfuse for prompt and LLM-level inspection |
| Evaluation | pytest-based harness, Ragas metrics, Langfuse datasets | Retrieval, trajectory, tool-call evals | Runs in CI and on sampled production traces |
| Guardrails | Presidio, Pydantic validators, local classifier model | PII, injection, output checks | Deterministic checks first, model checks second |

---

## 4. Modules

Modules marked **(§5)** implement a Protocol shown in §5. Their remaining methods are listed here.

| Module | Responsibility | Public interface | Benefit | Connects to |
|---|---|---|---|---|
| **Model Gateway** | Routes LLM calls by workspace policy (tier, privacy, cost, latency). Retry, fallback, token and cost metering. | `complete(req, route)`, `stream(req)`, `usage()` · wraps `LLMProvider` (§5) | Provider swap and failover without code change. Budget enforcement. | Planner, Reasoner, Guardrails, Observability |
| **Embedding Service** | Batch and query embeddings, model versioning per index. | `embed_documents()`, `embed_query()`, `model_id` · `EmbeddingModel` (§5) | Re-index on model change without touching retriever. | Ingestion, Retriever, Memory |
| **Retriever** | Dense + BM25 in parallel, RRF fusion, rerank, top-k with citations and namespace filters. | `retrieve(query, ctx, k)` · `RetrievalStrategy` (§5) | Better recall than dense-only, strategy per agent. | Embedding, VectorStore, Reranker, Reasoner |
| **Prompt Registry** | Versioned templates resolved per workspace/agent/sub-agent/node. Composes the sub-agent's persona template with the agent-level wizard block (Role, Language & constraints, Response format). Templates are general; slots (agent fields, tool block) are filled at run time. **Per-sub-agent seam: persona (§4.2).** | `resolve(name, ctx)` (`ctx` carries `agent_id`, `sub_agent_id`), `publish(draft)`, `pin(run)` · `PromptStore` (§5) | Prompt changes ship without deploys, with rollback and eval gating. A new sub-agent brings its own persona. | Planner, Reasoner, Responder, Evaluation |
| **Planner** | Turns a goal into a bounded plan over the active sub-agent's effective tools only: steps, dependencies, tool candidates. Caps on steps and time. Replans on failure. | `plan(state)`, `replan(state, failure)` | Prevents uncontrolled agent loops. Plans are inspectable. Cannot propose a tool the sub-agent does not hold. | Prompt Registry, Tool Registry, Model Gateway |
| **Tool Registry** | Holds the system catalog copy and computes each sub-agent's effective tool set from the request's `agent_config` (§4.1). Per tool: description, input schema (`parameters`), `result_summary` and `effect_summary` templates, `connector_id`, version, `deprecated`. `requires_approval` is the user's setting, sent in `agent_config`; the AI repo adds no policy floor. **Per-sub-agent seam: effective tool set (§4.2).** | `apply_catalog(snapshot)`, `effective_for(agent_ref, agent_config)`, `get(tool_id)`, `validate_args(call)`, `recompute(agent_config)` | Single definition feeds planner, policy and executor. Sub-agent sets cannot drift from the user's selection. | Sub-Agent Registry, Policy Engine, MCP Client, backend catalog push |
| **MCP Client** | Pooled MCP sessions, tool discovery, timeouts. Never handles credentials. A call to a connector that is not connected returns `connector_not_connected`. | `call_tool(connector_id, tool_id, args, idem_key)`, `list_tools(connector_id)` · adapts `Connector` (§5) | Connectors are decoupled from orchestration. | Executor, connectors (backend) |
| **Executor** | Runs authorized calls with an idempotency key. Backend handles transient provider errors and retries; emits `tool.*` events. | `execute(call) -> ToolOutcome`, `ledger.get(key)` | No duplicate executions on redelivery. | Tool Registry, MCP Client, Postgres |
| **Policy Engine** | Returns `allow`, `require_approval` or `deny` from allowlist (the active sub-agent's effective tool set), the tool's `requires_approval` flag (the user's setting), args (e.g. amount), actor. The AI repo adds no risk classification. | `evaluate(call, ctx) -> Decision` | Governance in one auditable place. Last line of defense against out-of-set calls. | Tool Registry, Executor, approval gate |
| **Memory** | Short-term window plus rolling summary. Long-term customer facts, workspace-scoped. | `read(scope, q)`, `write(scope, items)`, `summarize(thread)` · `MemoryStore` (§5) | Context continuity across turns and channels at bounded token cost. | Conversation Manager, Retriever, Embedding |
| **Conversation Manager** | Thread state and mode: Active / Assistance requested / Human control / Closed. Message ordering, job-to-thread mapping. Holds `Thread.active_sub_agent_id` (V1: constant). | `append(msg)`, `set_mode(mode, actor)`, `get_thread(id)` | Agent is silenced under human control and resumes cleanly. Thread ownership is ready for sub-agent pinning. | Orchestrator, Memory, API layer |
| **Ingestion Pipeline** | Fetch, parse, clean, chunk, embed, upsert. Files come from a presigned S3 URL. Status machine, content-hash dedupe. | `ingest(source)`, `retry(source_id, ref?)`, `delete(source_id)`, `purge(agent_id)`, `status(source_id)` | Idempotent, resumable, per-source failure isolation. | Parsers, Embedding, VectorStore, backend webhook |
| **Evaluation Harness** | Offline and CI suites: retrieval (Recall@k, MRR, nDCG), tool-call correctness, trajectory, groundedness. Samples production traces. | `run_suite(name, cfg)`, `compare(run_a, run_b)` | Every swap in §5 is gated by measured regression. | All modules, Langfuse, CI |
| **Guardrails** | Input: injection, PII. Output: groundedness, cross-workspace leakage, language, forbidden content. | `check_input(msg)`, `check_output(draft, ctx)` | Enforces the agent's constraints deterministically. | Orchestrator, Model Gateway, Retriever |
| **Sub-Agent Registry** (seam) | Loads `SubAgentSpec` manifests: `sub_agent_id`, audience, predefined tool ids, persona template refs. Resolves the `AgentRef` for a request. V1 ships one sub-agent: `customer_support`. | `get(sub_agent_id)`, `enabled(ctx)`, `resolve(req, ctx)` · `SubAgentRegistry`, `SubAgentResolver` (§5) | A new sub-agent is a manifest plus persona templates, with no change to existing modules. | Tool Registry, Prompt Registry, Conversation Manager, application layer |

### 4.1 Tool hierarchy and effective tool sets

Tool access is resolved through four levels. The user touches only the agent-level selection `S`.

| Level | Symbol | Owner | User-visible | Illustrative ids |
|---|---|---|---|---|
| System catalog | `C` | Backend, pushed to the AI repo; global (§4.3) | the user sees only tools of connected connectors | `[0,1,2,3,4,5,6,7]` |
| Sub-agent predefined set | `P_s` | AI repo, platform-owned manifest | **no** | `P(customer_support)=[0,2,4,6]`, `P(billing)=[0,1,3,5]` |
| Agent tool selection | `S ⊆ C` | User at agent creation (wizard step 3), persisted by the backend | yes | `[0,2,4,5]` |
| Sub-agent effective set | `E_s = P_s ∩ S` | Tool Registry, derived | **no**, never editable | `E(customer_support)=[0,2,4]`, `E(billing)=[0,5]` |

Propagation. The user edits only `S`, and every sub-agent's set follows:

| Agent selection `S` | `E(customer_support)`, `P=[0,2,4,6]` | `E(billing)`, `P=[0,1,3,5]` |
|---|---|---|
| `[0,2,4,5]` | `[0,2,4]` | `[0,5]` |
| `[0,2,5]` (user drops 4) | `[0,2]` | `[0,5]` |
| `[2,5]` (user drops 0) | `[2]` | `[5]` |
| `[0,1,2,3,4,5,6,7]` | `[0,2,4,6]` | `[0,1,3,5]` |

Tool 7 is in no predefined set, so no sub-agent can use it even when selected.

**Invariants**
1. `E_s = P_s ∩ S` is computed by `ToolRegistry.recompute()` from the request's `agent_config`. It is never an input field. No endpoint accepts `E_s`, and the user never edits a sub-agent's tool set.
2. Excluding a tool from `S` removes it from every sub-agent whose `P_s` contained it, with no further action. The allowlist is evaluated at call time, including on approval resume, using the `agent_config` sent with the decision (§6.5), so a removal also revokes pending approvals for that tool.
3. `P_s` is versioned with the code and never leaves the AI repo. `config:validate` returns only the derived `E_s`, and the backend must not render predefined sets.
4. `S ⊆ C` is checked against the system catalog only (the backend sends only tools of connected connectors). `config:validate` returns `unknown_tool` for an id missing from the catalog. At request time an unknown id triggers one `GET /catalog` reconcile. If it is still unknown, or the tool is deprecated, it is dropped from `E` with the event `config.warning` (`unknown_tool` or `tool_removed`) and the message is still answered. A selected id that is in no enabled sub-agent's `P_s` yields the warning `tool_unreachable`.
5. The `requires_approval` flag is per (agent, tool), applies to every sub-agent that holds the tool, and is set only by the user. It arrives inside `agent_config` as a boolean. The catalog has no approval flag. The AI repo does not model risk or a policy floor.
6. Every consumer (planner, `tool_selector`, policy, executor) obtains tools only through `effective_for(agent_ref, agent_config)`.
7. `effective_tools(P, S)` is a pure function in the domain layer. Property tests assert `E ⊆ P`, `E ⊆ S`, monotonicity in `S`, and idempotence. The tool-call eval suite asserts zero executed calls outside `E`.
8. Tool ids are stable and never reused, because the `P_s` manifests reference them. The backend deprecates a tool instead of deleting it, and a test checks that every `P_s` id exists in the catalog.
9. When the backend removes or deprecates a tool, it pushes the catalog first and then leaves the tool out of every `agent_config` it sends.

**V1 instantiation.** Only the `customer_support` sub-agent is enabled, so `E = P(customer_support) ∩ S`, where `S` comes from wizard step 3. The six catalog actions in the UI are in `P(customer_support)`. Refund order, when unchecked, is absent from `E` and cannot be planned, approved or executed.

### 4.2 Per-sub-agent seams

**Each sub-agent has its own persona prompt (Prompt Registry) and its own effective tool set (Tool Registry).** These are the two per-sub-agent seams. Agent-level settings (user instructions, knowledge, channels, selection `S`) are shared by all of the agent's sub-agents. Everything else in the runtime is shared.

| Seam | Where | Keyed by | V1 value | When a sub-agent is added |
|---|---|---|---|---|
| **Persona prompt** | Prompt Registry (`PromptStore`) | `(workspace, agent_id, sub_agent_id, node)` | `customer_support` persona template plus the agent-level wizard block (Role, Language & constraints, Response format) | Add persona templates for the new `sub_agent_id`. Resolution order in §2.3. |
| **Effective tool set** | Tool Registry | `AgentRef` | `P(customer_support) ∩ S` | Add a manifest with predefined ids. Sets derive per §4.1. |
| Identity | Domain `AgentRef(agent_id, sub_agent_id, audience)` | state, checkpoint, events, usage | `sub_agent_id` constant | none |
| Sub-agent manifest | `SubAgentRegistry` | `sub_agent_id` | one file manifest | add a manifest, enable per workspace in config |
| Resolution | `SubAgentResolver` | `agent_id` from request, channel, thread | `StaticResolver`: always `customer_support` | channel-to-audience strategy (§6.6) |
| Thread ownership | Conversation Manager | `Thread.active_sub_agent_id` | the one sub-agent | pins the selected sub-agent for the thread |
| Event and usage attribution | Event envelope, `usage.recorded` | `agent_id`, `sub_agent_id` | `sub_agent_id` constant | per-sub-agent analytics with no schema change |

### 4.3 System catalog and agent config

| | System catalog | Agent config |
|---|---|---|
| Scope | Global (all connectors) | One agent |
| Owner | Backend. The AI repo keeps a read-only copy in Postgres. | Backend only. The AI repo keeps a run-scoped copy for recovery and resume. |
| Delivery | `PUT /catalog` push, `GET /catalog` reconcile | `agent_config` inside every message, approval decision, preview and `config:validate` request |

- **Catalog:** the backend pushes the full catalog once per change, before the UI shows it. Delivery is at least once, and the AI repo updates its local catalog copy. The payload is one full snapshot, `{tools: [...]}`, never a per-tool patch. Fields per tool: `tool_id`, `connector_id`, `description`, `parameters` (the input JSON Schema), `result_summary` and `effect_summary` templates, `version`, `deprecated`.
- **Read-only vs. side-effecting:** `effect_summary` is `null` for a read-only tool and a deterministic template for a side-effecting tool. Preview mocks tools with a non-null `effect_summary` and runs `null` tools live (§7.1). The distinction does not change execution: every call carries an `idem_key` and goes through the ledger (§4 Executor).
- **Templates:** `result_summary` renders from the tool's result data and `effect_summary` from the call arguments. Both are deterministic `{field}` templates, never free LLM text (§8.2 step 6, §8.3).
- **Workspace and connectors:** backend-only, and the AI repo is not notified. A disconnect or a removed tool is handled by leaving those tools out of `agent_config`. A call that still reaches a disconnected connector returns `connector_not_connected` and hands off to a human.
- **Agent create and edit:** the backend stores the config and sends nothing to the AI repo. Only knowledge sources are sent when added (§8.5).
- **Each request:** the AI repo computes `E` (§4.1) from `agent_config` and keeps the run's copy in the job row and checkpoint. A config edit applies from the next message, and a catalog change applies immediately upon backend push.
- **Run time:** prompts are general templates. Slots come from `agent_config`, and the tool block comes from `E` plus the catalog descriptions.

---

## 5. Modularity and extensibility

**Rule:** every swappable component sits behind a Protocol in `ports/`. Swapping one means implementing the Protocol, registering the adapter, and changing config. Orchestration code is never edited.

Adding a sub-agent is not a swap. It is a manifest plus persona templates plus eval suites, and it needs the §6.6 node once a second sub-agent exists.

Enforcement:
- `import-linter` contract: `orchestrator/` may import only `ports/` and `domain/`.
- One shared contract test suite per Protocol, run against every adapter in CI.
- Property tests for `effective_tools` (§4.1).

### 5.1 Interface boundaries

```python
class LLMProvider(Protocol):  # openai | anthropic | gemini | local
    capabilities: ProviderCaps  # tools, json_mode, max_ctx

    async def complete(self, req: LLMRequest) -> LLMResponse: ...
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]: ...


class VectorStore(Protocol):  # qdrant | pgvector | weaviate | pinecone
    async def upsert(self, ns: NameSpace, items: list[VectorItem]) -> None: ...
    async def search(self, ns: NameSpace, q: HybridQuery, k: int) -> list[Hit]: ...
    async def delete(self, ns: NameSpace, flt: Filter) -> int: ...


class EmbeddingModel(Protocol):
    model_id: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...
    async def embed_query(self, text: str) -> Vector: ...


class PromptStore(Protocol):  # versioned, per-workspace, per-agent, per-sub-agent
    def resolve(
        self, name: str, ctx: PromptContext
    ) -> PromptVersion: ...  # ctx.agent_id, ctx.sub_agent_id


class RetrievalStrategy(Protocol):  # dense | hybrid_rrf | hybrid_rerank | graph (later)
    async def retrieve(self, q: Query, ctx: WorkspaceCtx) -> list[Chunk]: ...


class Connector(Protocol):  # MCP server, REST connector, native SDK
    async def list_tools(self, ctx: WorkspaceCtx) -> list[ToolSpec]: ...
    async def call(
        self, tool: str, args: dict, ctx: WorkspaceCtx, idem_key: str
    ) -> ToolResult: ...  # idem_key required for all calls


class MemoryStore(Protocol):  # postgres | redis | vector-backed
    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]: ...
    async def write(self, scope: MemoryScope, items: list[MemoryItem]) -> None: ...


class SubAgentRegistry(Protocol):  # file manifests (V1) | db-backed (later)
    def get(
        self, sub_agent_id: str
    ) -> SubAgentSpec: ...  # sub_agent_id, audience, predefined_tool_ids, persona refs
    def enabled(self, ctx: WorkspaceCtx) -> list[SubAgentSpec]: ...


class SubAgentResolver(Protocol):  # static (V1) | channel_audience (later)
    def resolve(
        self, req: ResolveRequest, ctx: WorkspaceCtx
    ) -> AgentRef: ...  # agent_id, channel, thread -> AgentRef
```

### 5.2 Selection by config

```yaml
workspace: acme
llm:        {provider: anthropic, tiers: {fast: <id>, standard: <id>, reasoning: <id>}, fallback: openai}
vector:     {store: qdrant, mode: shared}
embedding:  {model: bge-m3}
retrieval:  {strategy: hybrid_rerank, k: 8, rerank_k: 40}
memory:     {backend: postgres}
sub_agents: {enabled: [customer_support], resolver: static}
```

| Swap point | Implementations | Config key |
|---|---|---|
| LLM provider | openai, anthropic, gemini, vLLM | `llm.provider` |
| Vector store | qdrant, pgvector, weaviate, pinecone | `vector.store` |
| Embedding | bge-m3, OpenAI-compatible | `embedding.model` |
| Prompts | DB-backed, file-backed (dev) | `prompts.source` |
| Retriever | dense, hybrid_rrf, hybrid_rerank | `retrieval.strategy` |
| Tools | MCP, REST, native | per-tool `connector_id` |
| Memory | postgres, redis | `memory.backend` |
| Sub-agents and resolver | file manifests, static (V1); db manifests, channel_audience (later) | `sub_agents.enabled`, `sub_agents.resolver` |

### 5.3 Design patterns

| Pattern | Where |
|---|---|
| **Ports & Adapters** | `ports/` Protocols with `adapters/` per vendor. Orchestrator depends only on ports. |
| **Strategy** | `RetrievalStrategy`, model routing policy, chunking strategy per source type, sub-agent resolution policy (`SubAgentResolver`). |
| **Factory** | Registry builds adapters from config (`build_llm(cfg)`, `build_vector_store(cfg)`), cached per workspace config version. |
| **Adapter** | LiteLLM to `LLMProvider`. MCP session to `Connector`. Qdrant client to `VectorStore`. |
| **Repository** | Checkpoints, traces, idempotency ledger, prompt versions. Postgres hidden behind interfaces. |
| **Chain of Responsibility** | Policy rules (allowlist, requires_approval, amount threshold, rate limit). Guardrail checks. Gateway fallback chain. |

---

## 6. LangGraph state graph

### 6.1 State schema

```python
class AgentState(TypedDict):
    workspace_id: str
    agent_id: str  # the user-created agent, supplied by the backend
    sub_agent_id: str  # V1: "customer_support"; set at ingress, pinned
    audience: Literal["external", "internal"]  # V1: "external"
    channel: str  # whatsapp | widget | hosted | api (later: employee portal)
    conversation_id: str
    agent_config: AgentConfig  # run-scoped copy from the request
    messages: Annotated[list[Message], add_messages]
    plan: Plan | None  # steps, deps, cursor
    retrieved_context: list[Chunk]
    tool_calls: list[ToolCall]  # proposed + executed, with params_hash; idem_key on every call
    pending_approvals: list[Approval]
    execution_trace: Annotated[list[TraceEvent], operator.add]
    confidence: float
    handoff: HandoffReason | None
```

### 6.2 Nodes

| Node | Calls | Writes |
|---|---|---|
| `guardrails` (in/out) | Guardrails | `messages` (sanitized), `handoff` on violation |
| `router` | Conversation Manager, small LLM | route, `plan` stub |
| `planner` | Planner (persona and effective tools for `sub_agent_id`) | `plan` |
| `retriever` | Retriever | `retrieved_context` |
| `reasoner` | Model Gateway (persona for `sub_agent_id`) | draft, `confidence`, next action |
| `tool_selector` | Tool Registry `effective_for(agent_ref)` | `tool_calls` (args validated) |
| `policy_check` | Policy Engine (allowlist = sub-agent's effective set) | decision per call |
| `approval_gate` | `interrupt()` | `pending_approvals` |
| `executor` | Executor | `tool_calls[].outcome` |
| `responder` | Model Gateway, Prompt Registry | final message |
| `memory_writer` | Memory | long-term facts, thread summary |

No node is added for multi-sub-agent support. Every node resolves persona and tools from `(agent_id, sub_agent_id)` in state, so adding sub-agents does not change node code.

### 6.3 Graph

```mermaid
flowchart TD
  START([START]) --> GI["guardrails (input)"]
  %% SEAM (§6.6): a future sub_agent_selector node is inserted on the edge GI --> R. Not built in V1.
  GI -->|pass| R[router]
  GI -->|blocked| RS[responder]
  R -->|"mode = human_control"| END1([END: forward only])
  R -->|"intent = question"| RT[retriever]
  R -->|"intent = task"| PL[planner]
  R -->|"greeting / needs clarification"| RS
  PL -->|"step needs knowledge"| RT
  PL -->|"next step is tool call"| TS[tool_selector]
  RT --> RE[reasoner]
  RE -->|"needs tool"| TS
  RE -->|"answer ready (confidence < τ sets handoff=low_confidence)"| RS
  TS --> PC[policy_check]
  PC -->|allow| EX[executor]
  PC -->|require_approval| AG[approval_gate]
  PC -->|deny| RS
  AG -->|approved| EX
  AG -->|"rejected / expired"| RS
  EX -->|"succeeded, steps remain"| PL
  EX -->|"succeeded, plan done"| RS
  EX -->|"failed (handoff=tool_failed)"| RS
  RS --> GO["guardrails (output)"]
  GO -->|pass| MW[memory_writer]
  GO -->|"violation, first time"| RS
  MW --> END2([END])
```

### 6.4 Edges

| From → To | Condition |
|---|---|
| START → guardrails(in) | always |
| guardrails(in) → router / responder | pass / blocked (refusal text). **The pass edge is the sub-agent selection insertion point (§6.6).** |
| router → END | conversation mode is Human control; message is forwarded, no agent reply |
| router → retriever / planner / responder | intent is question / task / greeting or ambiguous |
| planner → retriever / tool_selector | next plan step is knowledge / tool |
| retriever → reasoner | always |
| reasoner → tool_selector / responder | model requests a tool / answer ready |
| reasoner → responder (`handoff=low_confidence`) | `confidence < τ` (per-agent) |
| tool_selector → policy_check | always, one pass per call |
| policy_check → executor / approval_gate / responder | allow / approval required / deny |
| approval_gate → executor / responder | approved / rejected, expired, revoked authority, or tool removed from the effective set |
| executor → planner / responder | succeeded with steps left / plan done |
| executor → responder (`handoff=tool_failed`) | execution failed |
| responder → guardrails(out) → memory_writer → END | pass. On violation: regenerate once, then fixed safe fallback. |

### 6.5 Conditional routing

1. **Approval pause and resume.** `policy_check` returns `require_approval`. `approval_gate` persists an immutable params snapshot, emits `approval.required`, and calls `interrupt()`. The job becomes `waiting_approval` and the run's background task ends. `POST /workspaces/{workspace_id}/approvals/{id}/decision` carries the current `agent_config` and starts a new background task that replaces the run's copy with it and resumes the thread with `Command(resume={decision, approver_id})`. On resume the gate checks `params_hash`, TTL and that the tool is still in the sub-agent's effective tool set before routing to `executor`.
2. **Tool execution failure.** When the Backend returns a failure status (exhausted retries, provider error, or verification failure on the Backend), route to `responder` with `handoff=tool_failed`. Conversation mode becomes Assistance requested. The agent never claims success.
3. **Low confidence.** `confidence < τ`, or a retrieval miss on a policy-sensitive topic, routes to `responder` with a handoff message. A human takes over with the plan, retrieved context and actions already taken attached.

### 6.6 Extension point: sub-agent selection (documented, not built in V1)

**Position:** the edge `guardrails(in) → router`. Selection runs after input sanitization and before intent routing, because router, planner and reasoner behave differently per sub-agent.

```text
V1:      START → guardrails(in) → router → …                                 AgentRef resolved at ingress
Future:  START → guardrails(in) → sub_agent_selector → router → …            sub_agent_id resolved in-graph
```

| Aspect | V1 | Future `sub_agent_selector` node |
|---|---|---|
| Who resolves `AgentRef` | The submit-message use case calls `SubAgentResolver` (`StaticResolver`: always `customer_support`) and writes `agent_id`, `sub_agent_id`, `audience` into initial state | The node calls the same `SubAgentResolver` and writes `sub_agent_id` and `audience`. `agent_id` is already in state. |
| Inputs | `agent_id` from the request | `channel`, channel-to-audience map (config), `Thread.active_sub_agent_id`, sanitized message |
| Routing | none | External channels (WhatsApp, chat widget, hosted page, Custom UI API) route to the agent's external sub-agents. Internal channels (employee portal) route to its internal sub-agents (HR, IT). |

Rules for the future node:
- Routing policy lives behind `SubAgentResolver` (Strategy). The node only calls it and writes state.
- The node selects only `sub_agent_id`. `agent_id` always comes from the backend's channel binding and is never chosen or changed by the AI repo.
- Audience is derived from `channel` via config, never from message content, so prompt injection cannot flip an external conversation to an internal sub-agent.
- Selection runs on fresh turns only. `resume` jobs restore a checkpoint with `agent_ref` already pinned and skip it.
- Once selected, `Thread.active_sub_agent_id` pins the sub-agent for the thread. Switching sub-agents mid-thread is an explicit handoff and out of scope.
- Downstream nodes are unchanged because persona and tools are resolved from state (§4.2).
- Adoption is three edits: register the node, replace the one edge, and stop resolving `sub_agent_id` at ingress.

---

## 7. AI ↔ Backend API contract

**Schema choice**
- **REST + JSON (OpenAPI)** for config and control. Synchronous, debuggable, and generates typed clients for the backend.
- **Async job pattern** for plan execution: `202 + job_id`, then a webhook. Plans exceed gateway timeouts and may pause for hours on approval.
- **SSE** for token and trace streaming. It is one-way, works through standard proxies, and reconnects with `Last-Event-ID`. Control actions stay on REST, so WebSocket's bidirectionality adds nothing.
- The backend proxies SSE to clients. WhatsApp receives `message.final` only.
- The public `/v1/agents/{id}/conversations` in the docs screen is backend-owned and maps to the contract below.

**Workspace context:** one `get_workspace_context()` dependency reads `workspace_id` from the `/workspaces/{workspace_id}/` URL path parameter. Every AI-service request is scoped under this path, except the catalog endpoints (`/catalog`), which are global. The backend stores agent configs: the AI repo has no endpoint to create or read one, and every request that runs or validates an agent carries its `agent_config`.

### 7.1 Endpoints

In every path below, `{workspace_id}` is the workspace id, and `{id}` under `/agents/` is the `agent_id`.

| Method | Path | Request | Response | Async |
|---|---|---|---|---|
| POST | `/workspaces/{workspace_id}/agents/{id}/config:validate` | `{agent_config}` (wizard draft) | `{valid, errors[], warnings[]}` | no |
| POST | `/workspaces/{workspace_id}/agents/{id}/preview` | `{message, agent_config}`; tools with a non-null `effect_summary` are mocked, tools with `effect_summary: null` run live | SSE stream | stream |
| PUT | `/catalog` | `{tools[]}`, the full catalog (§4.3) | `{applied}` | no |
| GET | `/catalog` | n/a (`?full=true` returns the tools) | `{tools[]}` | no |
| POST | `/workspaces/{workspace_id}/knowledge-sources` | `{agent_id, type: url\|text\|txt\|pdf\|docx, ref, name}`. `ref` is a presigned S3 URL for files. No source id is sent: the AI repo creates it and the backend stores it. | `202 {source_id, status: processing}` | yes |
| GET | `/workspaces/{workspace_id}/knowledge-sources/{id}` | n/a | `{status, chunks, error?}` | no |
| POST | `/workspaces/{workspace_id}/knowledge-sources/{id}/retry` | `{ref?}` (a fresh URL if the old one expired) | `202` | yes |
| DELETE | `/workspaces/{workspace_id}/knowledge-sources/{id}` | n/a | `204` (purges vectors) | no |
| DELETE | `/workspaces/{workspace_id}/agents/{id}/knowledge-sources` | n/a | `204` (purges the agent's sources) | no |
| POST | `/workspaces/{workspace_id}/conversations/{id}/messages` | Header `Idempotency-Key`; `{agent_id, channel, sender, content, locale, agent_config}` | `202 {job_id}` | **yes** |
| GET | `/workspaces/{workspace_id}/conversations/{id}/events` | Header `Last-Event-ID` | SSE stream (§7.2) | stream |
| GET | `/workspaces/{workspace_id}/jobs/{job_id}` | n/a | `{status: queued\|running\|waiting_approval\|done\|failed}` | no |
| GET | `/workspaces/{workspace_id}/conversations/{id}/trace` | n/a | Ordered trace events | no |
| POST | `/workspaces/{workspace_id}/conversations/{id}/mode` | `{mode: human_control\|active\|closed, actor}` | `200` | no |
| POST | `/workspaces/{workspace_id}/approvals/{id}/decision` | `{decision, approver_id, note?, agent_config}` | `202 {job_id}` | yes |
| POST | `{backend}/webhooks/ai` | `job.completed`, `approval.required`, `ingestion.status`, `usage.recorded` | `2xx` | n/a |

**`agent_config`.** Sent by the backend in every request that runs or validates an agent. Required: `name`, `instructions` (role, language_constraints, response_format), `tools[{tool_id, requires_approval}]`. Optional: `description`, `sub_agents`, and any other agent-level parameter (unknown fields are ignored). It applies to that request only.

Backend-owned lists (conversations, approval queue and history, execution history) are projections of AI events. The AI repo exposes no list endpoints for them.

**Additive seam fields.** These are optional, V1 clients may ignore them, and no path, method or existing field changes:
- `agent_config`: its `tools` list is the agent's tool selection `S` with its approval flags (§4.1). Optional `sub_agents` (list of `sub_agent_id`) defaults to `["customer_support"]`. Any other value returns the error `sub_agent_unsupported`.
- `config:validate` response: optional `effective_tools` (derived `E_s`, a map `{sub_agent_id: [tool ids]}`), the warning codes `tool_unreachable` and `tool_removed`, and the error code `unknown_tool`. Predefined sets are never returned.
- `POST /workspaces/{workspace_id}/conversations/{id}/messages`: the request carries `agent_config`, and a repeated `Idempotency-Key` returns the stored job and ignores a changed one. `agent_id` identifies the agent. The backend never sends `sub_agent_id`, because the AI repo resolves it (§6.6).
- Event envelope: `agent_id`, `sub_agent_id` (§7.2).

### 7.2 Event vocabulary

Envelope: `{event_id, seq, ts, type, workspace_id, agent_id, sub_agent_id, conversation_id, job_id, data}`. `seq` is monotonic per conversation.

| Event | Emitted by | `data` keys |
|---|---|---|
| `plan.created` | planner | steps[] |
| `retrieval.completed` | retriever | sources[], chunk_count |
| `config.warning` | tool registry | code (`unknown_tool` \| `tool_removed`), tool_ids[] |
| `tool.started` | executor | call_id, tool_id, tool_version, action, integration, params_summary |
| `tool.completed` | executor | call_id, status, result_summary, initiated_by (`auto` \| member id) |
| `approval.required` | approval_gate | approval_id, action, integration, params, params_hash, effect_summary, expires_at |
| `approval.resolved` | approval_gate | approval_id, decision, approver_id |
| `handoff.requested` | responder | reason |
| `message.delta` | responder | text |
| `message.final` | responder | message_id, text, citations[] |
| `usage.recorded` | gateway, executor | tokens, cost, executions |
| `error` | any | code, retryable, message |

---

## 8. End-to-end workflow traced through the UI

### 8.1 Create Agent wizard

| Step | UI | AI repo stores | AI repo validates |
|---|---|---|---|
| 1. Name & Instructions | Agent name, Role, Language & constraints, Response format | Nothing. The backend stores name and instructions and sends them in `agent_config`; they fill the template slots at run time, composed with the `customer_support` sub-agent persona | Token budget, supported languages (English, Arabic), injection patterns in free text, contradictions with platform-level rules |
| 2. Knowledge | Website URL, Text snippet, TXT, PDF, DOCX; per-source status | Source records, chunks and vectors under `NameSpace(workspace, agent, source)`. The backend sends each source to `POST /workspaces/{workspace_id}/knowledge-sources` (S3 URL for files) | Reachability, mime and size limits, content-hash duplicates. Continue allowed while Processing; a Failed source becomes a review warning. |
| 3. Actions | Six actions with checkbox and Approval toggle | Nothing. The backend stores `S` and the per-tool approval flag and sends them in `agent_config.tools`. Tool Registry derives `E(customer_support) = P(customer_support) ∩ S` at run time (§4.1). | `config:validate`: every `tool_id` exists in the catalog (`unknown_tool`). The AI repo does not change `requires_approval`. |
| 4. Channels | Chat widget, Hosted page, Custom UI (API), WhatsApp | Nothing. The channel binding is backend-owned, and each message carries `channel` | Channel capabilities: streaming on widget, hosted page and API; none on WhatsApp (final message only) |
| 5. Preview & Review | Counts: sources and chunks, allowed actions, approvals required, channel status | Nothing | `config:validate` returns readiness report, with action counts computed from `E` (source counts come from `ingestion.status` webhooks). `/workspaces/{workspace_id}/agents/{id}/preview` runs the real graph with tools that have a non-null `effect_summary` mocked and `effect_summary: null` tools live. |
| Activate | Activate agent | Nothing | Backend only: checks that its sources are `ready` and flips the status. The first message carries `agent_config`. |

Draft persistence ("Save draft") is backend-side. AI validation is stateless. Agent creation is not an AI event: the AI repo first hears of an agent from a source, a validate or preview call, or a message. The V1 wizard has no sub-agent picker: every agent has the one `customer_support` sub-agent, and the backend may omit `sub_agents`.

### 8.2 Agent runtime: Conversation #1042, Sara Ali

| # | UI behavior | AI repo behavior | Event |
|---|---|---|---|
| 1 | Sara asks "Where is my order #1042?" via WhatsApp, chat widget or hosted page | Backend posts the message with `agent_id` and `agent_config`. AI returns `202 + job_id`. The use case resolves `AgentRef` (`agent_id` from the request, `sub_agent_id` = `customer_support`, external) into state. Guardrails pass, router sees task intent. | n/a |
| 2 | Agent replies "Let me check…" | Planner builds the plan: lookup order, then answer. Retriever pulls shipping-policy chunks from Customer Experience sources. | `plan.created`, `retrieval.completed` |
| 3 | Trace row "Looked up order · Shopify · Done" | `tool_selector` picks `Lookup order` from the sub-agent's effective set (Custom API, Shopify, `GET /orders/1042`). Policy: allow, no approval. Executor calls via MCP. | `tool.started`, `tool.completed` |
| 4 | Agent: shipped, tracking #TRK8812 | Reasoner grounds the reply in the tool result. Output guardrails pass. | `message.delta`, `message.final` |
| 5 | Sara: "send me the invoice" | New turn. Plan: `Send invoice` (QuickBooks), params invoice #, amount, recipient email. | `plan.created` |
| 6 | Card: "Action is waiting for approval" with Approve / Reject. Trace row "Send invoice · QuickBooks · Pending". | `policy_check`: the allowlist rule passes, then the approval rule returns `require_approval`. Gate snapshots params, renders `effect_summary` from a deterministic template (never free LLM text), calls `interrupt()`. | `approval.required` |
| 7 | Agent tells Sara an authorized member will approve | Responder emits the holding message. Job state `waiting_approval`. | `message.final` |
| 8 | Authorized member approves in Approvals | Backend checks the "Approve actions" permission, then calls `/workspaces/{workspace_id}/approvals/{id}/decision` with the current `agent_config`. Graph resumes. Gate verifies `params_hash`, TTL and effective-set membership. | `approval.resolved` |
| 9 | Execution history shows "Invoice INV-2210 sent", initiated by the approver | Executor runs with the ledger key. `initiated_by` is the approver's id. | `tool.started`, `tool.completed` |
| 10 | Trace shows Done | Backend verifies invoice creation and returns success with invoice details. Failure triggers §6.5 case 2. | `tool.completed` |
| 11 | Sara receives the invoice confirmation | Responder emits `message.final` as a proactive turn on the same thread. Backend delivers it through the channel adapter. | `message.final`, `job.completed` |

Takeover, as in "Human takeover — Omar Farouk joined": `POST /workspaces/{workspace_id}/conversations/{id}/mode` sets Human control. The router stops agent replies, and the AI keeps appending inbound messages to the thread for context.

### 8.3 Approvals queue

| UI | AI repo |
|---|---|
| Pending: action, integration, conversation, customer, age | Backend projects from `approval.required`. |
| Detail: requested action, integration, originating conversation, requester ("Customer Experience on behalf of Sara Ali"), parameters, effect sentence | All fields come from the frozen payload. The effect sentence is `effect_summary`. |
| Approve / Reject | Resume with decision. Reject routes to `responder` ("execution stopped") and records `Rejected` in the trace. |
| History: decision, decider, time | Backend persists from `approval.resolved`. The AI trace keeps the decision and `approver_id`. |

### 8.4 Execution history

Each row maps to one `tool.completed` event:
- **Action, Integration:** action definition.
- **Result:** `result_summary`, a per-tool template from the catalog such as "Email sent to {recipient}".
- **Initiated by:** `auto` or the approver's id.
- **Time:** event `ts`.
- **Rejected:** the executor is never invoked. `approval.resolved` with `decision=rejected` is projected as "Rejected — execution stopped".
- Dashboard counters (executions 24h, failed and retried) are backend aggregates over these events plus `error`.

### 8.5 Knowledge ingestion

```mermaid
flowchart LR
  A["Source added<br/>url / snippet / TXT / PDF / DOCX"] --> B["Processing"]
  B --> C["Fetch + parse"] --> D["Clean + chunk"] --> E["Embed"] --> F["Upsert dense + sparse"]
  F --> G["Ready: chunk count"]
  C -->|error| X["Failed"]
  E -->|error| X
  X -->|"POST retry"| B
```

- **Fetch:** files come from the presigned S3 URL in `ref`, and the AI repo has no S3 credentials. File hosts are allowlisted, and website fetches block private address ranges.
- **Chunking:** structure-aware, with section headings carried in chunk metadata. Arabic text is normalized (diacritics, alef/yā variants) before the BM25 leg. The source's content hash makes ingestion idempotent.
- **Status:** `ingestion.status` webhooks carry `source_id`, `agent_id`, `processing`, `ready`, `failed`, `chunks` and an error code (`fetch_failed` for an expired or failed URL).
- **Retry:** `POST /workspaces/{workspace_id}/knowledge-sources/{id}/retry` is the "Retry available" action on a failed DOCX like `onboarding-guide.docx`. After `fetch_failed` the backend retries with a fresh `ref`.
- **Delete:** purges vectors by `source_id` (`DELETE /workspaces/{workspace_id}/knowledge-sources/{id}`). `DELETE /workspaces/{workspace_id}/agents/{id}/knowledge-sources` purges all sources of a deleted agent.

---

## 9. Future features

- **Multiple sub-agents per agent (support, billing, sales, …):** each new sub-agent is a `SubAgentSpec` manifest (predefined tool ids), persona templates and eval suites. Add the `sub_agent_selector` node at the §6.6 edge. Effective tool sets follow from §4.1 with no extra configuration. Add sub-agent-specific nodes only if a sub-agent needs a different topology.
- **Internal employee sub-agents (HR, IT, …):** `audience=internal` sub-agents, selected by channel through the same §6.6 node, with identity-aware retrieval and employee-scoped actions (see the portals bullet).
- **Neo4j knowledge graph:** multi-hop reasoning (customer → invoice → contract → ticket). Per-workspace configurable ontology. A `graph` `RetrievalStrategy` added behind the existing port, with no orchestrator change.
- **Temporal.io (Durable Execution Engine).** Introduces Temporal to handle long-running, scheduled, and time-bound enterprise workflows, augmenting LangGraph (which handles the cognitive steps).
  - **Durable Timers & SLAs:** Native long-lived timers (days/weeks) for SLA detection (e.g., escalating a ticket open past a threshold) without holding worker resources.
  - **Scheduled & Recurring Workflows:** Native cron-like scheduling for recurring agent runs (e.g., monthly billing reviews, contract renewals).
  - **Guaranteed Execution:** Automatic state recovery and infinite retries for external system calls, ensuring side effects complete even during long infrastructure failures.
  - **Architecture Shift:** LangGraph remains the cognitive orchestrator (LLM calls, tool selection); Temporal wraps the overall process, managing durable state, multi-day pauses, and scheduled triggers.
- **Internal company portals:**
  - Provisioning portal for companies to create and manage agents for their own clients. This needs hierarchical workspaces (`parent_workspace_id`), inherited policy and model config, and cross-client isolation tests.
  - Employee portal with identity-aware retrieval (ACL-filtered knowledge) and employee-scoped actions.
- **Voice agent channel:** streaming STT and TTS adapters in front of the same graph, barge-in handling, and a latency budget for the `interactive` jobs.

---

## 10. Non-goals

- **Multiple sub-agents and internal sub-agents (V1):** each agent ships with one `customer_support` sub-agent. The seams exist (§4.2, §6.6), but there is no selection node, no second sub-agent, and no cross-sub-agent handoff.
- **User-editable sub-agent tool sets:** the user selects only the agent-level subset `S`. Predefined sets are platform-owned, and effective sets are derived.
- **Authentication and authorization:** the backend authenticates users and enforces roles. The AI repo trusts the workspace context the backend supplies (`/workspaces/{workspace_id}/` path parameter) and enforces only agent-level policy.
- **Billing and plan limits:** the AI repo emits `usage.recorded`. Metering, quotas and invoicing are backend.
- **Multi-workspace config persistence:** the AI repo keeps no workspace, member or agent source of truth, only derived artifacts.
- **UI:** no dashboards, builders, approval screens or notification rendering. The AI repo emits events and the frontend renders them.
- **Connector and OAuth flows:** OAuth, token refresh, credential storage and connector or MCP server implementations (Odoo included) are backend. The AI repo is an MCP client. It is not notified when a workspace connects or disconnects a connector and never receives credentials.
- **AI-initiated runs:** every V1 run starts from a request that carries `agent_config`. A run started by the AI repo itself (SLA timers, scheduled follow-ups) needs a backend call to fetch the config and is not part of V1.
- **Channel delivery:** WhatsApp, widget and hosted-page transport and webhooks are backend. The AI repo emits messages.
- **Mobile app:** out of scope for the AI repo. It consumes the same events through the backend.
