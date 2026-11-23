# Felops AI

AI service for Felops customer-support agents. The backend owns auth, workspace config and channels. This repo handles the agent runtime: orchestration, retrieval, tool calls and approvals.

The service is being built in phases. Everything below is what runs today, not the full design.

## What works now

P0 foundations, tasks 0.1 to 0.5:

- Repo skeleton with `domain/`, `ports/`, `adapters/` and `orchestrator/`. Import-linter enforces that the orchestrator imports only ports and domain. CI runs the linter, the architecture check and the tests.
- Docker compose starts the API, Postgres, Qdrant and a stub MCP server.
- Domain types: `WorkspaceContext`, `NameSpace`, `AgentRef`, `AgentState`, the event envelope and the retrieval value objects.
- Ports with contract tests: `LLMProvider`, `PromptStore`, `VectorStore` and `EmbeddingModel`. The tests run against in-memory stubs in `tests/fakes/`.

Not built yet: real adapters, graph nodes, the message API and streaming. Those start at task 0.6.

## Run it

```bash
cp .env.example .env
make up
curl localhost:8000/health
```

`make up` builds the images and waits for every container to report healthy. `make logs` follows the output, `make down` stops everything and `make reset` also drops the volumes.

The health check returns `{"status":"ok"}`.

## Develop

Python 3.12. Tests run offline and need no containers.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/lint-imports
```

## Layout

```
src/felops/
  domain/        pure types, no I/O
  ports/         protocols, the swap points
  services/      module logic
  orchestrator/  LangGraph graph and nodes
  adapters/      vendor code
  app/           composition root
  api/           FastAPI transport
tests/
  unit/          domain, ports and API tests
  fakes/         in-memory stubs for contract tests
```

## Docs

- [Design and architecture](docs/new_version/ai_system_design_and_architecture.md)
- [Execution plan](docs/new_version/Felops%20AI%20Repo%20%E2%80%94%20Execution%20Plan.md)
- [AI service contract](docs/new_version/ai-service-contract.md)

The design doc has the full architecture. This file only tracks what runs.
