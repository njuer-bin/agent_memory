"""P2 neighboring-context expansion regression tests.

Run:
    G:\\aconda\\python.exe tests\\p2_test.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def test_fact_expands_adjacent_raw_context():
    with tempfile.TemporaryDirectory() as d:
        engine = MemoryEngine(f"{d}/memory.db")
        user = "p2-context"

        messages = [
            AddMessage(role="user", content="我最近准备搬家。", timestamp=1000),
            AddMessage(role="user", content="我现在住在上海。", timestamp=2000),
            AddMessage(role="user", content="公司安排我下个月去上海工作。", timestamp=3000),
            AddMessage(role="user", content="所以我会搬过去。", timestamp=4000),
        ]
        engine.add(AddRequest(
            request_id="p2-context-001",
            messages=messages,
            user_id=user,
            session_id="p2-session",
        ))

        results = engine.search(SearchRequest(
            query="我现在住在哪里？",
            user_id=user,
            top_k=5,
        ))

        assert any(r["content"] == "我现在住在上海。" for r in results)
        assert any(
            r["content"] == "我最近准备搬家。"
            and (r.get("metadata") or {}).get("context_expanded")
            for r in results
        )
        assert any(
            r["content"] == "公司安排我下个月去上海工作。"
            and (r.get("metadata") or {}).get("context_expanded")
            for r in results
        )


def test_context_expansion_respects_user_isolation():
    with tempfile.TemporaryDirectory() as d:
        engine = MemoryEngine(f"{d}/memory.db")

        engine.add(AddRequest(
            request_id="p2-user-a",
            messages=[AddMessage(role="user", content="我现在住在上海。", timestamp=1000)],
            user_id="p2-a",
            session_id="s-a",
        ))
        engine.add(AddRequest(
            request_id="p2-user-b",
            messages=[AddMessage(role="user", content="我现在住在深圳。", timestamp=1000)],
            user_id="p2-b",
            session_id="s-b",
        ))

        results = engine.search(SearchRequest(
            query="我现在住在哪里？",
            user_id="p2-a",
            top_k=5,
        ))

        assert any("上海" in r["content"] for r in results)
        assert not any("深圳" in r["content"] for r in results)


def main():
    tests = [
        test_fact_expands_adjacent_raw_context,
        test_context_expansion_respects_user_isolation,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"P2 TEST PASSED: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
