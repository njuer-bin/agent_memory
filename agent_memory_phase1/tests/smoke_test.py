import os
import sys
import uuid
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")

headers = {}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

def check(resp, expected):
    print(resp.request.method, resp.request.url, resp.status_code, resp.text)
    assert resp.status_code == expected, resp.text

def main():
    check(requests.get(f"{BASE}/health", timeout=10), 200)

    rid = "smoke-" + uuid.uuid4().hex
    payload = {
        "request_id": rid,
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "我现在住在杭州，我不喜欢早班飞机。"
            }
        ],
        "user_id": "smoke-user-001",
        "session_id": "smoke-session-001",
    }

    # First Add.
    r1 = requests.post(f"{BASE}/add", json=payload, headers=headers, timeout=10)
    check(r1, 200)
    assert r1.json()["success"] is True
    assert r1.json()["request_id"] == rid

    # Second Add with same request_id must be idempotent.
    r2 = requests.post(f"{BASE}/add", json=payload, headers=headers, timeout=10)
    check(r2, 200)
    assert r2.json()["request_id"] == rid

    # Search must be user-isolated.
    s = {
        "query": "杭州",
        "user_id": "smoke-user-001",
        "top_k": 5,
    }
    r3 = requests.post(f"{BASE}/search", json=s, headers=headers, timeout=10)
    check(r3, 200)
    results = r3.json()["results"]
    assert results, "expected at least one result"
    assert all(x["user_id"] == "smoke-user-001" for x in results)

    # Another user must not see it.
    s2 = {"query": "杭州", "user_id": "smoke-user-002", "top_k": 5}
    r4 = requests.post(f"{BASE}/search", json=s2, headers=headers, timeout=10)
    check(r4, 200)
    assert r4.json()["results"] == []

    print("\nSMOKE TEST PASSED")

if __name__ == "__main__":
    main()
