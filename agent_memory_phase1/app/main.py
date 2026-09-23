from fastapi import FastAPI
from .db import init_db, count_memories
from .routes import router

app = FastAPI(
    title="Agent Memory Challenge - Phase 1",
    version="0.1.0",
)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "service": "agent-memory",
        "version": "0.1.0",
        "memories": count_memories(),
    }

app.include_router(router)
