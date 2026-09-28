"""Strict AML API contract regression test."""

from __future__ import annotations

import os
import uuid

import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30


def main():
    user = "contract-" + uuid.uuid4().hex[:10]
    session = "contract-session"
    request_id = "contract-" + uuid.uuid4().hex

    add_payload = {
        "request_id": request_id,
        "user_id": user,
        "session_id": session,
        "messages": [{"role": "user", "content": "我现在住在杭州。"}],
    }

    add = requests.post(BASE + "/add", json=add_payload, timeout=TIMEOUT)
    assert add.status_code == 200, (add.status_code, add.text)
    assert add.json() == {
        "success": True,
        "request_id": request_id,
        "user_id": user,
        "session_id": session,
    }

    search = requests.post(
        BASE + "/search",
        json={"query": "我现在住哪里？", "user_id": user, "top_k": 100},
        timeout=TIMEOUT,
    )
    assert search.status_code == 200, (search.status_code, search.text)
    body = search.json()
    assert set(body) == {"data"}
    assert len(body["data"]) <= 100

    for item in body["data"]:
        assert set(item) == {"id", "content", "score", "created_at"}

    missing_query = requests.post(
        BASE + "/search",
        json={"user_id": user, "top_k": 100},
        timeout=TIMEOUT,
    )
    assert missing_query.status_code == 422

    missing_top_k = requests.post(
        BASE + "/search",
        json={"query": "我现在住哪里？", "user_id": user},
        timeout=TIMEOUT,
    )
    assert missing_top_k.status_code == 422

    print("AML API CONTRACT TEST PASSED")


if __name__ == "__main__":
    main()
