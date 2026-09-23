from typing import List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(min_length=1)
    timestamp: Optional[int] = None
    content: str = Field(min_length=1)

class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: List[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)

class AddResponse(BaseModel):
    success: bool
    request_id: str

class MemoryEvidence(BaseModel):
    id: str
    content: str
    role: str
    timestamp: Optional[int] = None
    user_id: str
    session_id: str
    score: float
    source: str = "raw"

class SearchResponse(BaseModel):
    results: List[MemoryEvidence]
