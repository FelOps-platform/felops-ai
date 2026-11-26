from contextlib import asynccontextmanager

from fastapi import FastAPI

from felops.api.routes.messages import router as messages_router
from felops.app.container import init_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = init_container()
    app.state.container = container
    app.state.graph = container.graph
    yield
    # Shutdown: release P3 resources (Postgres checkpointer, connection pools) here.
    app.state.container = None
    app.state.graph = None


app = FastAPI(title="Felops AI", lifespan=lifespan)
app.include_router(messages_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
