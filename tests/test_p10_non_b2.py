import os
from types import SimpleNamespace

os.environ["AML_USE_LLM"] = "0"

from memory_engine.p10_non_b2 import _calendar_bounds, _current_state, _session_summary, _b2


class FakeStore:
    def __init__(self):
        self.facts = [
            {"id": "old", "user_id": "u", "subject": "user", "predicate": "residence", "object": "南京", "content": "我住南京", "timestamp": 100, "valid_from": 100, "status": "active"},
            {"id": "new", "user_id": "u", "subject": "user", "predicate": "residence", "object": "上海", "content": "我现在住上海", "timestamp": 200, "valid_from": 200, "status": "active"},
        ]
        self.raw = [
            {"id": "r1", "user_id": "u", "session_id": "s1", "timestamp": 100, "content": "我参加了摄影活动"},
            {"id": "r2", "user_id": "u", "session_id": "s1", "timestamp": 200, "content": "后来我开始学习剪辑"},
            {"id": "r3", "user_id": "u", "session_id": "s2", "timestamp": 300, "content": "我完成了一个项目"},
        ]

    def active_facts(self, user_id, include_history=False):
        return list(self.facts)

    def all_raw(self, user_id, session_id=None):
        rows = list(self.raw)
        if session_id is not None:
            rows = [x for x in rows if x["session_id"] == session_id]
        return sorted(rows, key=lambda x: x["timestamp"], reverse=True)


def test_calendar_last_year_is_precise():
    ref = 1735689600000  # 2025-01-01 UTC
    start, end = _calendar_bounds("去年发生了什么", ref)
    assert start is not None and end is not None
    assert start < end
    assert start == 1704067200000


def test_current_state_prefers_latest_active_value():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(user_id="u", query="我现在住哪里", include_history=False)
    plan = SimpleNamespace(predicate_hint="residence")
    rows = _current_state(engine, request, plan)
    assert rows[0]["metadata"]["object"] == "上海"
    assert rows[0]["metadata"]["state_rank"] == 1


def test_session_summary_keeps_multiple_sessions():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(user_id="u", query="总结一下我的历史", include_history=True)
    rows = _session_summary(engine, request)
    assert rows
    assert any("s1" in row["content"] for row in rows)
    assert any("s2" in row["content"] for row in rows)


def test_b2_is_strictly_bypassed():
    plan = SimpleNamespace(multi_hop=True)
    assert _b2("为什么他后来离职", plan) is True
    assert _b2("他现在在哪家公司", plan) is False
