"""AML Phase 2 competition-oriented integration test.

Run with the service already started:
    G:\\aconda\\python.exe tests\\competition_test.py

Optional:
    set BASE_URL=http://127.0.0.1:8000
    set MEMORY_API_KEY=your-secret

The test creates isolated temporary user IDs and does not rely on the
repository's runtime database contents.
"""

from __future__ import annotations

import os
import time
import uuid

import requests


BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("MEMORY_API_KEY", "").strip()
TIMEOUT = 30
RUN = uuid.uuid4().hex[:10]
USER = f"competition-{RUN}"
OTHER = f"competition-other-{RUN}"


def request(method, path, **kwargs):
    kwargs.setdefault("timeout", TIMEOUT)
    return requests.request(method, BASE + path, **kwargs)


def add(content, *, user=USER, request_id=None, messages=None, timestamp=None):
    if messages is None:
        messages = [{
            "role": "user",
            "content": content,
            **({"timestamp": timestamp} if timestamp is not None else {}),
        }]
    rid = request_id or ("competition-" + uuid.uuid4().hex)
    r = request("POST", "/add", json={
        "request_id": rid,
        "messages": messages,
        "user_id": user,
        "session_id": "competition-session",
    })
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()


def search(query, *, user=USER, **extra):
    payload = {"query": query, "user_id": user, "top_k": 10}
    payload.update(extra)
    r = request("POST", "/search", json=payload)
    assert r.status_code == 200, (r.status_code, r.text)
    return r.json()["results"]


def contains(results, text):
    return any(text in item.get("content", "") for item in results)


def test_health():
    r = request("GET", "/health", timeout=10)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_basic_add_search():
    add("我现在住在杭州。")
    assert contains(search("我现在住哪里？"), "杭州")


def test_idempotency():
    rid = "competition-idempotent-" + RUN
    add("我的固定测试偏好是咖啡。", request_id=rid)
    add("我的固定测试偏好是咖啡。", request_id=rid)
    results = search("我的固定测试偏好是什么？")
    assert sum("咖啡" in x.get("content", "") for x in results) >= 1


def test_immediate_consistency():
    add("我刚刚记录了一个立即可检索事实：喜欢跑步。")
    assert contains(search("我喜欢什么运动？"), "跑步")


def test_user_isolation():
    add("我住在深圳。", user=OTHER)
    results = search("我现在住哪里？", user=OTHER)
    assert contains(results, "深圳")
    assert not contains(results, "杭州")


def test_current_history():
    add("我以前住在南京。", timestamp=1704067200000)
    add("我现在住在上海。", timestamp=1735689600000)
    current = search("我现在住哪里？")
    history = search("我以前住哪里？", include_history=True)
    assert contains(current, "上海")
    assert contains(history, "南京")


def test_relation_multihop():
    add("我的朋友小王。")
    add("小王推荐我去杭州。")
    results = search("朋友推荐的城市是什么？", multi_hop=True)
    assert contains(results, "杭州")


def test_event_rule_temporal():
    add("2024年6月我参加了上海科技展。", timestamp=1719792000000)
    add("请记住，我通常周末喝咖啡。", timestamp=1735689600000)
    assert contains(search("我参加了什么展会？"), "上海科技展")
    assert contains(search("我的习惯是什么？"), "周末喝咖啡")


def test_memory_filters():
    add("我的职业是软件工程师。")
    results = search("我的职业是什么？", memory_types=["fact"])
    assert results
    assert any("软件工程师" in x.get("content", "") for x in results)


def test_session_filter():
    add("这是 session-a 的记忆。")
    rid = "competition-session-b-" + RUN
    r = request("POST", "/add", json={
        "request_id": rid,
        "messages": [{"role": "user", "content": "这是 session-b 的记忆。"}],
        "user_id": USER,
        "session_id": "session-b",
    })
    assert r.status_code == 200
    results = search("session-a 的记忆", session_id="competition-session")
    assert contains(results, "session-a")


def test_twenty_message_boundary():
    messages = [
        {"role": "user", "content": f"20-message-boundary-{i}"}
        for i in range(20)
    ]
    add("", messages=messages)
    assert search("20-message-boundary-19")


def test_over_twenty_messages():
    messages = [
        {"role": "user", "content": f"over-20-boundary-{i}"}
        for i in range(21)
    ]
    add("", messages=messages)
    assert search("over-20-boundary-20")


def test_large_message():
    text = "large-message-token " * 5000
    add(text)
    assert search("large-message-token")


def test_malformed_payloads():
    r = request("POST", "/add", json={"request_id": "bad"})
    assert r.status_code in (400, 422)

    r = request("POST", "/search", json={"user_id": USER})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_add_search_latency():
    t0 = time.perf_counter()
    results = search("我现在住哪里？")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert results
    print(f"single search latency: {elapsed_ms:.2f} ms")


def test_auth_if_configured():
    if not API_KEY:
        print("auth test skipped: MEMORY_API_KEY is not configured")
        return

    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = request("POST", "/search", headers=headers,
                json={"query": "我现在住哪里？", "user_id": USER, "top_k": 3})
    assert r.status_code == 200

    bad = request("POST", "/search",
                  headers={"Authorization": "Bearer wrong-key"},
                  json={"query": "我现在住哪里？", "user_id": USER, "top_k": 3})
    assert bad.status_code == 401


def main():
    tests = [
        test_health,
        test_basic_add_search,
        test_idempotency,
        test_immediate_consistency,
        test_user_isolation,
        test_current_history,
        test_relation_multihop,
        test_event_rule_temporal,
        test_memory_filters,
        test_session_filter,
        test_twenty_message_boundary,
        test_over_twenty_messages,
        test_large_message,
        test_malformed_payloads,
        test_add_search_latency,
        test_auth_if_configured,
    ]

    print("=" * 72)
    print("AML PHASE 2 COMPETITION TEST")
    print(f"API: {BASE}")
    print("=" * 72)

    passed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            raise

    print("=" * 72)
    print(f"COMPETITION TEST PASSED: {passed}/{len(tests)}")
    print("COMPETITION READY")


if __name__ == "__main__":
    main()
