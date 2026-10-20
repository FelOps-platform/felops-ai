from fastapi import FastAPI

app = FastAPI(title="Felops AI")


@app.get("/health")
async def health():
    return {"status": "ok"}