from contextlib import asynccontextmanager

from fastapi import FastAPI

from felops.app.container import init_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = init_container()
    app.state.container = container
    app.state.graph = container.graph
    app.state.embedding = container.embedding
    app.state.vector_store = container.vector_store
    try:
        yield
    finally:
        # Shutdown: release adapters and P3 resources here.
        try:
            close_fn = getattr(container.embedding, "close", None)
            if callable(close_fn):
                await close_fn()
        finally:
            try:
                close_fn = getattr(container.vector_store, "close", None)
                if callable(close_fn):
                    await close_fn()
            finally:
                app.state.container = None
                app.state.graph = None
                app.state.embedding = None
                app.state.vector_store = None


app = FastAPI(title="Felops AI", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
