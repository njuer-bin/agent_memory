from fastapi import APIRouter, Depends, HTTPException
from .auth import require_auth
from .db import add_memory, search_memory
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse

router = APIRouter()

@router.post("/add", response_model=AddResponse)
async def add(req: AddRequest, _=Depends(require_auth)):
    # Pydantic validates required fields and non-empty content.
    # Idempotency is implemented in the DB by PRIMARY KEY(request_id).
    try:
        add_memory(req)
        return AddResponse(success=True, request_id=req.request_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"add failed: {e}")

@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, _=Depends(require_auth)):
    try:
        return SearchResponse(
            results=search_memory(req.user_id, req.query, req.top_k)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")
