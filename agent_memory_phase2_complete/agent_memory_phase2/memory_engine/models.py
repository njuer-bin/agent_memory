from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def fingerprint(*parts: Any) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AddMessage(BaseModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool
    request_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: Optional[str] = None
    question: Optional[str] = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    session_id: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    memory_types: Optional[list[str]] = None
    include_history: bool = False
    multi_hop: Optional[bool] = None


class SearchResult(BaseModel):
    id: str
    content: str
    role: str
    timestamp: int
    user_id: str
    session_id: str
    score: float
    source: str
    memory_type: str
    status: str = "active"
    valid_from: Optional[int] = None
    valid_to: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult]
