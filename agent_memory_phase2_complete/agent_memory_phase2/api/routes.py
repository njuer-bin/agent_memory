from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddRequest, AddResponse, SearchRequest, SearchResponse, SearchResult

app = FastAPI(title="AML Phase 2 Memory Engine", version="2.0.0")
engine = MemoryEngine(os.getenv("MEMORY_DB_PATH", "data/memory.db"))

API_KEY = os.getenv("MEMORY_API_KEY", "").strip()


def check_auth(authorization: str | None, x_api_key: str | None):
    if not API_KEY:
        return
    token = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif authorization:
        token = authorization.strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="invalid credentials")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add", response_model=AddResponse)
def add(
    payload: AddRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    check_auth(authorization, x_api_key)
    try:
        engine.add(payload)
        return AddResponse(success=True, request_id=payload.request_id)
    except Exception as exc:
        # 不吞掉异常；接口返回明确 500，便于平台重试。
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    check_auth(authorization, x_api_key)
    try:
        rows = engine.search(payload)
        results = [SearchResult(**r) for r in rows]
        return SearchResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
