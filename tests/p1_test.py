"""P1 governance and retrieval regression tests.

Run:
    G:\\aconda\\python.exe tests\\p1_test.py
"""

from __future__ import annotations

import tempfile

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddRequest, AddMessage, SearchRequest


def add(engine, user_id, content, role="user", timestamp=None):
    engine.add(AddRequest(
        request_id=f"p1-{user_id}-{role}-{abs(hash((content, timestamp))) }",
        messages=[AddMessage(role=role, content=content, timestamp=timestamp)],
        user_id=user_id,
        session_id="p1-session",
    ))


def test_agent_fact_conflict_is_audited():
    with tempfile.TemporaryDirectory() as d:
        engine = MemoryEngine(f"{d}/memory.db")
        user = "p1-conflict"

        add(engine, user, "我现在住在上海。", role="user", timestamp=1000)
        add(engine, user, "我现在住在北京。", role="assistant", timestamp=2000)

        logs = engine.store.conflict_logs(user)
        assert logs
        assert logs[0]["resolution"] == "preserved_lower_authority"
        assert logs[0]["old_object"] == "上海"
        assert logs[0]["new_object"] == "北京"

        facts = engine.store.active_facts(user, include_history=True)
        current = [x for x in facts if x["status"] == "active" and x["object"] == "上海"]
        conflicted = [x for x in facts if x["conflict_status"] == "conflict" and x["object"] == "北京"]
        assert current
        assert conflicted
        assert conflicted[0]["source"] == "assistant"


def test_new_user_fact_supersedes_agent_fact():
    with tempfile.TemporaryDirectory() as d:
        engine = MemoryEngine(f"{d}/memory.db")
        user = "p1-resolution"

        add(engine, user, "我现在住在北京。", role="assistant", timestamp=1000)
        add(engine, user, "我现在住在上海。", role="user", timestamp=2000)

        logs = engine.store.conflict_logs(user)
        assert logs
        assert logs[0]["resolution"] == "resolved_by_newer_source"

        facts = engine.store.active_facts(user, include_history=True)
        assert any(x["status"] == "active" and x["object"] == "上海" for x in facts)
        assert any(x["status"] == "superseded" and x["object"] == "北京" for x in facts)


def test_controlled_second_round_for_multihop():
    with tempfile.TemporaryDirectory() as d:
        engine = MemoryEngine(f"{d}/memory.db")
        user = "p1-multihop"

        add(engine, user, "我的朋友小王。")
        add(engine, user, "小王推荐我去杭州。")

        results = engine.search(SearchRequest(
            query="朋友推荐的城市是什么？",
            user_id=user,
            top_k=5,
            multi_hop=True,
        ))
        assert any("杭州" in x["content"] for x in results)


def main():
    tests = [
        test_agent_fact_conflict_is_audited,
        test_new_user_fact_supersedes_agent_fact,
        test_controlled_second_round_for_multihop,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"P1 TEST PASSED: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
