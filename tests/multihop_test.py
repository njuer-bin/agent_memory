import os
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
USER = "multi-hop-user"


def add(rid, content):
    r = requests.post(BASE + "/add", json={
        "request_id": rid,
        "messages": [{"role": "user", "timestamp": 1704067200000, "content": content}],
        "user_id": USER,
        "session_id": "mh",
    })
    r.raise_for_status()


def main():
    add("mh-1", "我的朋友小王。")
    add("mh-2", "我朋友小王推荐我去杭州。")

    r = requests.post(BASE + "/search", json={
        "query": "朋友推荐的城市是什么？",
        "user_id": USER,
        "top_k": 5,
    })
    r.raise_for_status()
    data = r.json()
    rows = data.get("data", [])
    assert "data" in data, f"unexpected /search response: {data}"
    assert rows
    assert any("杭州" in x["content"] for x in rows)

    # AML-style three-hop regression:
    # friend -> workplace -> company headquarters.
    add("mh-3", "小王在 Acme 工作。")
    add("mh-4", "Acme 总部在上海。")

    r = requests.post(BASE + "/search", json={
        "query": "朋友工作的公司总部在哪里？",
        "user_id": USER,
        "top_k": 5,
    })
    r.raise_for_status()
    data = r.json()
    rows = data.get("data", [])
    assert "data" in data, f"unexpected /search response: {data}"
    assert rows
    assert any("上海" in x["content"] for x in rows), rows

    print("MULTI-HOP TEST PASSED")


if __name__ == "__main__":
    main()
