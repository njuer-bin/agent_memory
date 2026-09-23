from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    api_key: str = os.getenv("API_KEY", "")
    db_path: str = os.getenv("DB_PATH", "./data/memory.db")
    max_retries_add: int = 32
    retry_after_max_seconds: int = 60
    request_timeout_seconds: int = 30 * 60

settings = Settings()
