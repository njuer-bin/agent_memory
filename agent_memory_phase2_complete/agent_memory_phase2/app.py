from __future__ import annotations

import logging
import os

# The memory engine uses logger.info() for optional profiling. Configure the
# root logger here because app.py is the normal production entry point and
# Uvicorn does not guarantee that application INFO logs are visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from api.routes import app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
