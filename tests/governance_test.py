import os
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
USER = "governance-user"

def add(rid, content, ts):
    r = requests.post(BASE + "/add", json={
        "request_id": rid,
        "messages": [{"role":"user","timestamp":ts,"content":content}],
        "user_id": USER,
        "session_id": "s1",
    })
    r.raise_for_status()

def search(q, history=False):
    r = requests.post(BASE + "/search", json={
        "query":q,"user_id":USER,"top_k":10,"include_history":history
    })
    r.raise_for_status()
    return r.json()["results"]

def main():
    add("gov-1", "我现在住在杭州。", 1704067200000)
    add("gov-2", "我现在住在上海。", 1704153600000)

    current = search("我现在住哪里？")
    assert current
    assert any("上海" in x["content"] for x in current)

    history = search("我以前住哪里？", history=True)
    assert history
    assert any("杭州" in x["content"] for x in history)

    print("GOVERNANCE TEST PASSED")

if __name__ == "__main__":
    main()
