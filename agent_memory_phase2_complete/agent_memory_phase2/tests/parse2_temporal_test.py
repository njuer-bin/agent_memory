import os
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def add(user, rid, content, ts, session="parse2"):
    r = requests.post(BASE + "/add", json={
        "request_id": rid,
        "messages": [{"role": "user", "timestamp": ts, "content": content}],
        "user_id": user,
        "session_id": session,
    })
    r.raise_for_status()


def search(user, q, history=False, multi_hop=False):
    r = requests.post(BASE + "/search", json={
        "query": q,
        "user_id": user,
        "top_k": 10,
        "include_history": history,
        "multi_hop": multi_hop,
    })
    r.raise_for_status()
    return r.json()["results"]


def main():
    user = "parse2-temporal-user"
    t1 = 1704067200000       # 2024-01-01
    t2 = 1735689600000       # 2025-01-01

    add(user, "p2-1", "我现在住在杭州，我喜欢咖啡。", t1)
    add(user, "p2-2", "后来我搬到上海。", t2)

    current = search(user, "我现在住哪里？")
    assert current and any("上海" in x["content"] for x in current)

    history = search(user, "我以前住哪里？", history=True)
    assert history and any("杭州" in x["content"] for x in history)

    add(user, "p2-3", "我的朋友小王推荐我去杭州。", t2)
    hops = search(user, "朋友推荐的城市是什么？", multi_hop=True)
    assert hops and any("杭州" in x["content"] for x in hops)

    add(user, "p2-4", "请记住，我通常周末喝咖啡。", t2)
    rules = search(user, "我的习惯是什么？")
    assert rules and any("周末喝咖啡" in x["content"] for x in rules)

    print("PARSE2 TEMPORAL TEST PASSED")


if __name__ == "__main__":
    main()
