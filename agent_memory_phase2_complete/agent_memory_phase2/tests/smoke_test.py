import json
import os
import sys
import uuid
from pathlib import Path

import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
USER = "smoke-user"
SESSION = "smoke-session"
RID = "smoke-" + uuid.uuid4().hex

def post(path, payload):
    r = requests.post(BASE + path, json=payload, timeout=30)
    print(path, r.status_code, r.text)
    r.raise_for_status()
    return r.json()

def main():
    r = requests.get(BASE + "/health", timeout=10)
    print("GET /health", r.status_code, r.text)
    assert r.ok

    payload = {
        "request_id": RID,
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "我现在住在杭州，我喜欢咖啡。"
            }
        ],
        "user_id": USER,
        "session_id": SESSION,
    }

    a = post("/add", payload)
    assert a["success"] is True
    assert a["request_id"] == RID

    # 相同 request_id 必须幂等。
    a2 = post("/add", payload)
    assert a2["success"] is True
    assert a2["request_id"] == RID

    s = post("/search", {
        "query": "我现在住哪里？",
        "user_id": USER,
        "top_k": 5,
    })
    assert "results" in s
    assert len(s["results"]) >= 1

    # 严格用户隔离。
    s2 = post("/search", {
        "query": "我现在住哪里？",
        "user_id": "another-user",
        "top_k": 5,
    })
    assert s2["results"] == []

    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
