import os
import uuid

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
    r.raise_for_status()

    payload = {
        "request_id": RID,
        "messages": [{
            "role": "user",
            "timestamp": 1704067200000,
            "content": "我现在住在杭州，我喜欢咖啡。",
        }],
        "user_id": USER,
        "session_id": SESSION,
    }

    a = post("/add", payload)
    assert a == {
        "success": True,
        "request_id": RID,
        "user_id": USER,
        "session_id": SESSION,
    }

    # 相同 request_id 必须幂等。
    a2 = post("/add", payload)
    assert a2 == a

    s = post("/search", {
        "query": "我现在住哪里？",
        "user_id": USER,
        "top_k": 5,
    })
    assert "data" in s
    assert "results" not in s
    assert len(s["data"]) >= 1
    assert set(s["data"][0]) == {"id", "content", "score", "created_at"}

    # 严格用户隔离。
    s2 = post("/search", {
        "query": "我现在住哪里？",
        "user_id": "another-user",
        "top_k": 5,
    })
    assert s2["data"] == []

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
