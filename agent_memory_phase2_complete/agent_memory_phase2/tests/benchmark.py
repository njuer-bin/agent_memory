"""AML Phase-2 local benchmark harness.

Usage:
    G:\\aconda\\python.exe tests\\benchmark.py

Optional:
    set BASE_URL=http://127.0.0.1:8000
    set BENCHMARK_TOP_K=10
"""

from __future__ import annotations

import os
import statistics
import time
import uuid

import requests


BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
TOP_K = int(os.getenv("BENCHMARK_TOP_K", "10"))
TIMEOUT = 30

T = {
    "old": 1704067200000,
    "mid": 1719792000000,
    "new": 1735689600000,
}


def post(path, payload):
    r = requests.post(BASE + path, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def add(user, content, ts, session):
    post("/add", {
        "request_id": "bench-" + uuid.uuid4().hex,
        "messages": [{
            "role": "user",
            "timestamp": ts,
            "content": content,
        }],
        "user_id": user,
        "session_id": session,
    })


def search(user, query, **kwargs):
    payload = {"query": query, "user_id": user, "top_k": TOP_K}
    payload.update(kwargs)
    return post("/search", payload)["results"]


def rank_of(results, expected):
    expected = expected.lower()
    for i, r in enumerate(results, 1):
        if expected in r["content"].lower():
            return i
    return None


def timed_search(user, query, **kwargs):
    t0 = time.perf_counter()
    results = search(user, query, **kwargs)
    return results, (time.perf_counter() - t0) * 1000


def setup():
    add("bench-a", "我现在住在杭州。", T["old"], "session-a")
    add("bench-a", "后来我搬到上海。", T["new"], "session-a")
    add("bench-a", "我喜欢咖啡。", T["mid"], "session-a")
    add("bench-a", "我不喜欢太甜的饮料。", T["mid"], "session-a")
    add("bench-a", "我的朋友小王。", T["mid"], "session-a")
    add("bench-a", "我的朋友小王推荐我去杭州。", T["mid"], "session-a")
    add("bench-a", "2024年3月我去了北京参加展会。", T["mid"], "session-a")
    add("bench-a", "请记住，我通常周末喝咖啡。", T["new"], "session-a")
    add("bench-a", "我现在住在上海。", T["new"], "session-a")

    # Cross-user isolation sentinel.
    add("bench-b", "我住在深圳。", T["new"], "session-b")


def run_case(name, user, query, expected, **kwargs):
    results, latency = timed_search(user, query, **kwargs)
    rank = rank_of(results, expected)
    return {
        "name": name,
        "pass": rank is not None,
        "rank": rank,
        "latency_ms": latency,
        "top": results[0]["content"] if results else "",
    }


def main():
    r = requests.get(BASE + "/health", timeout=10)
    r.raise_for_status()

    print("=" * 72)
    print("AML PHASE-2 MEMORY BENCHMARK")
    print("=" * 72)
    print(f"BASE={BASE}  TOP_K={TOP_K}")

    setup()

    cases = [
        ("explicit_current_fact", "bench-a", "我现在住哪里？", "上海", {}),
        ("historical_fact", "bench-a", "我以前住哪里？", "杭州",
         {"include_history": True}),
        ("preference", "bench-a", "我喜欢什么？", "咖啡", {}),
        ("multi_hop", "bench-a", "朋友推荐的城市是什么？", "杭州",
         {"multi_hop": True}),
        ("rule", "bench-a", "我的习惯是什么？", "周末喝咖啡", {}),
        ("event", "bench-a", "我去北京参加了什么？", "北京", {}),
        ("isolation", "bench-b", "我现在住哪里？", "深圳", {}),
    ]

    rows = []
    for name, user, query, expected, kwargs in cases:
        rows.append(run_case(name, user, query, expected, **kwargs))

    # Stronger isolation assertion: the other user's Shanghai must not appear.
    leak_results = search("bench-b", "我现在住哪里？")
    isolation_ok = (
        any("深圳" in r["content"] for r in leak_results)
        and not any("上海" in r["content"] for r in leak_results)
    )

    print()
    print(f"{'CASE':28} {'PASS':6} {'RANK':6} {'LAT(ms)':10}")
    print("-" * 56)
    for row in rows:
        print(f"{row['name']:28} {str(row['pass']):6} "
              f"{str(row['rank'] or '-'):6} {row['latency_ms']:10.2f}")

    passed = sum(r["pass"] for r in rows)
    total = len(rows)
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    mrr = statistics.mean(1.0 / x for x in ranks) if ranks else 0.0
    recall = passed / total if total else 0.0
    latencies = [r["latency_ms"] for r in rows]

    print()
    print("METRICS")
    print(f"Recall@{TOP_K}: {recall:.3f} ({passed}/{total})")
    print(f"MRR@{TOP_K}:    {mrr:.3f}")
    print(f"Search p50:     {statistics.median(latencies):.2f} ms")
    print(f"Search mean:    {statistics.mean(latencies):.2f} ms")
    print(f"Isolation:      {'PASS' if isolation_ok else 'FAIL'}")

    failed = [r["name"] for r in rows if not r["pass"]]
    if failed or not isolation_ok:
        print()
        print("FAILED CASES:", ", ".join(failed) if failed else "none")
        if not isolation_ok:
            print("FAILED CASE: cross-user isolation")
        raise SystemExit(1)

    print()
    print("BENCHMARK PASSED")


if __name__ == "__main__":
    main()
