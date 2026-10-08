from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException

from memory_engine.reasoning_engine import ReasoningMemoryEngine
from memory_engine.models import AddRequest, AddResponse, SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

app = FastAPI(title="AML Phase 2 Memory Engine", version="2.0.0")
engine = ReasoningMemoryEngine(os.getenv("MEMORY_DB_PATH", "data/memory.db"))

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


def _created_at(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


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
        # AML requires these values to be returned exactly as supplied.
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )
    except Exception as exc:
        logger.exception("ADD_FAILED request_id=%s user_id=%s", payload.request_id, payload.user_id)
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
        # Keep the internal engine schema private. AML Search returns only:
        # id, content, score, created_at, wrapped in {"data": [...]}.
        results = [
            SearchResult(
                id=row["id"],
                content=row["content"],
                score=float(row["score"]),
                created_at=_created_at(int(row.get("timestamp", 0))),
            )
            for row in rows
        ]
        return SearchResponse(data=results)
    except Exception as exc:
        logger.exception("SEARCH_FAILED user_id=%s", payload.user_id)
        raise HTTPException(status_code=500, detail=str(exc))
