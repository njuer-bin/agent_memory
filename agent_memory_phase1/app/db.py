import hashlib
import sqlite3
import threading
import uuid
from pathlib import Path
from .config import settings

_lock = threading.RLock()

def connect():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _lock, connect() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp INTEGER,
            content TEXT NOT NULL,
            content_norm TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(request_id) REFERENCES requests(request_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_user
            ON memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_user_ts
            ON memories(user_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_memories_request
            ON memories(request_id);
        """)

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def token_score(query: str, content: str) -> float:
    q = normalize(query)
    c = normalize(content)
    if not q or not c:
        return 0.0

    # Lightweight lexical score for Phase 1.
    # Phase 2 will replace this with sparse+dense hybrid retrieval.
    q_chars = set(q)
    overlap = sum(1 for ch in q_chars if ch in c)
    char_score = overlap / max(1, len(q_chars))

    q_terms = q.split()
    term_hits = sum(1 for t in q_terms if t and t in c)
    term_score = term_hits / max(1, len(q_terms))

    exact_bonus = 1.0 if q in c else 0.0
    return min(1.0, 0.45 * char_score + 0.45 * term_score + 0.10 * exact_bonus)

def add_memory(req):
    import time
    with _lock, connect() as conn:
        existing = conn.execute(
            "SELECT request_id FROM requests WHERE request_id=?",
            (req.request_id,)
        ).fetchone()
        if existing:
            return False

        conn.execute(
            "INSERT INTO requests(request_id,user_id,session_id,created_at) VALUES(?,?,?,?)",
            (req.request_id, req.user_id, req.session_id, time.time())
        )

        for msg in req.messages:
            conn.execute(
                """INSERT INTO memories
                (id,request_id,user_id,session_id,role,timestamp,content,content_norm,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    req.request_id,
                    req.user_id,
                    req.session_id,
                    msg.role,
                    msg.timestamp,
                    msg.content,
                    normalize(msg.content),
                    time.time(),
                )
            )
    return True

def search_memory(user_id: str, query: str, top_k: int):
    with _lock, connect() as conn:
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE user_id=?
               ORDER BY CASE WHEN timestamp IS NULL THEN 0 ELSE timestamp END DESC,
                        created_at DESC""",
            (user_id,)
        ).fetchall()

    scored = []
    for row in rows:
        score = token_score(query, row["content"])
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: (-x[0], -(x[1]["timestamp"] or 0), -x[1]["created_at"]))

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "role": row["role"],
            "timestamp": row["timestamp"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "score": round(float(score), 6),
            "source": "raw",
        }
        for score, row in scored[:top_k]
    ]

def count_memories():
    with _lock, connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
