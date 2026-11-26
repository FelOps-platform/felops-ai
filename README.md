# Felops AI

AI service for Felops customer-support agents. The backend owns auth, workspace config and channels. This repo handles the agent runtime: orchestration, retrieval, tool calls and approvals.

The service is being built in phases. Everything below is what runs today, not the full design.

## What works now

Phase 0 foundations, tasks 0.1 to 0.10:

- **0.1 Repo skeleton and quality gates:** `domain/`, `ports/`, `services/`, `orchestrator/`, `adapters/`, `app/`, `api/` and `observability/`. CI runs `ruff check`, the import-linter architecture contracts (Design §5 layers and orchestrator boundaries) and `pytest` on `dev` and `main`.
- **0.2 Docker Compose stack:** `api`, `postgres`, `qdrant`, `mock-mcp` and `mock-backend` with healthchecks, plus `make up`, `down`, `logs` and `reset`.
- **0.3 Domain types:** `WorkspaceContext`, `NameSpace`, `AgentRef`, `AgentConfig`, `AgentState`, `EventEnvelope`, `MemoryItem`/`MemoryScope`, and the retrieval and tool value objects.
- **0.4–0.6 All six port protocols and contract tests:** `LLMProvider`, `PromptStore`, `VectorStore`, `EmbeddingModel`, `Connector` and `MemoryStore`, verified against in-memory fakes in `tests/fakes/`.
- **0.7 Real LLM and prompt adapters:** `LiteLLMProvider` (`src/felops/adapters/llm/litellm.py`) and `FilePromptStore` (`src/felops/adapters/prompts/file_store.py`) with templates in `prompts/platform/` (`default.md`, `router.md`, `responder.md`). `services/model_gateway.py` and `config/default.yaml` remain placeholders.
- **0.8 Mock backend (`mocks/mock_backend/`):** catalog endpoints (`GET /catalog`, `POST /catalog/push`), `agent_config` fixtures, KB file serving and a webhook sink.
- **0.9 Walking skeleton graph and FastAPI shell:** `START -> router -> responder -> END` with `InMemorySaver` checkpointing and `make_thread_config`, initialized in the FastAPI lifespan via `app/container.py`. It exposes `GET /health` and emits a Phase 0 canned stub reply (`router` and `responder` do not call `LLMProvider` yet).
- **0.10 Fixtures (`fixtures/`):** bilingual AR+EN KB documents (`fixtures/kb/`), 20 labelled Q&A pairs (`fixtures/qa/qa_seed.jsonl`) and 6 catalog tool entries (`fixtures/catalog/catalog.json`).

Not built yet (Phase 1+): `POST /workspaces/{workspace_id}/conversations/{conversation_id}/messages`, SSE `/events` streaming, LLM-backed `router`/`retriever`/`reasoner`/`responder`, `ModelGateway` routing/fallback, and real `VectorStore`/`EmbeddingModel`/`MemoryStore` adapters. (`McpConnector` and `mock-mcp` are already in the stack for tool connectivity.)

## Run it

```bash
cp .env.example .env  # optional: Compose has local defaults; needed for API_KEY or custom credentials
make up
curl localhost:8000/health
```

`make up` builds the images and waits for every container to report healthy. `make logs` follows the output, `make down` stops everything and `make reset` also drops the volumes.

The health check returns `{"status":"ok"}`.

## Develop

Python 3.12. Tests run offline and need no containers.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"   # or: uv sync --frozen --extra dev
.venv/bin/pytest -m "not live"      # offline tests when .env has an API_KEY
.venv/bin/ruff check .
.venv/bin/lint-imports
docker compose --env-file /dev/null config --quiet
```

## Layout

```
src/felops/
  domain/         pure types, no I/O
  ports/          protocols, the swap points
  services/       module logic
  orchestrator/   LangGraph graph and nodes
  adapters/       vendor code
  app/            composition root
  api/            FastAPI transport
  observability/  tracing and metrics (Phase 0 placeholder)
config/           default settings and subagent config
prompts/          platform and customer-support prompt templates
fixtures/         KB, Q&A and catalog fixtures
mocks/            mock backend and mock MCP servers
tests/
  unit/           adapters, api, domain, mocks, orchestrator and ports tests
  fakes/          in-memory stubs for contract tests
```

## Docs

- [Design and architecture](docs/new_version/ai_system_design_and_architecture.md)
- [Execution plan](docs/new_version/Felops%20AI%20Repo%20%E2%80%94%20Execution%20Plan.md)
- [AI service contract](docs/new_version/ai-service-contract.md)

The design doc has the full architecture. This file only tracks what runs.
