from fastapi import Header, HTTPException
from .config import settings

async def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    # Empty API_KEY means local/dev mode.
    if not settings.api_key:
        return

    if x_api_key == settings.api_key:
        return

    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == settings.api_key:
            return
        if authorization == settings.api_key:
            return

    raise HTTPException(status_code=401, detail="invalid credentials")
