import os

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest
from memory_engine.temporal_parser import normalize_temporal


def add(engine, rid, content, ts=None):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p9-user",
        session_id="p9-session",
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_last_year_is_calendar_anchored():
    reference = 1735689600000  # 2025-01-01 UTC
    info = normalize_temporal("去年我住在哪里？", reference)
    assert info.start == 1704067200000
    assert info.end == 1735689599999
    assert info.relation == "before"


def test_current_state_returns_active_value(tmp_path):
    old = os.environ.get("AML_USE_LLM")
    os.environ["AML_USE_LLM"] = "0"
    try:
        engine = MemoryEngine(str(tmp_path / "memory.db"))
        add(engine, "p9-current-1", "我住在南京。", 1704067200000)
        add(engine, "p9-current-2", "我住在上海。", 1735689600000)
        rows = engine.search(SearchRequest(query="我现在住哪里？", user_id="p9-user", top_k=5))
        assert rows
        assert "上海" in rows[0]["content"]
    finally:
        if old is None:
            os.environ.pop("AML_USE_LLM", None)
        else:
            os.environ["AML_USE_LLM"] = old


def test_rule_retrieval_is_explicit(tmp_path):
    old = os.environ.get("AML_USE_LLM")
    os.environ["AML_USE_LLM"] = "0"
    try:
        engine = MemoryEngine(str(tmp_path / "memory.db"))
        add(engine, "p9-rule-1", "请记住，我通常周末喝咖啡。", 1735689600000)
        rows = engine.search(SearchRequest(query="我的习惯是什么？", user_id="p9-user", top_k=5))
        assert any("周末喝咖啡" in row["content"] for row in rows)
    finally:
        if old is None:
            os.environ.pop("AML_USE_LLM", None)
        else:
            os.environ["AML_USE_LLM"] = old


def test_summary_candidate_covers_multiple_memory_types(tmp_path):
    old = os.environ.get("AML_USE_LLM")
    os.environ["AML_USE_LLM"] = "0"
    try:
        engine = MemoryEngine(str(tmp_path / "memory.db"))
        add(engine, "p9-s-1", "我的职业是软件工程师。")
        add(engine, "p9-s-2", "2024年我参加了上海科技展。", 1704067200000)
        rows = engine.search(SearchRequest(query="请总结一下你记得的我的经历。", user_id="p9-user", top_k=5))
        summary = [row for row in rows if row.get("memory_type") == "summary"]
        assert summary
        assert "软件工程师" in summary[0]["content"]
        assert "上海科技展" in summary[0]["content"]
    finally:
        if old is None:
            os.environ.pop("AML_USE_LLM", None)
        else:
            os.environ["AML_USE_LLM"] = old


def test_unrequested_sensitive_memory_is_deprioritized(tmp_path):
    old = os.environ.get("AML_USE_LLM")
    os.environ["AML_USE_LLM"] = "0"
    try:
        engine = MemoryEngine(str(tmp_path / "memory.db"))
        add(engine, "p9-h-1", "我的手机号是13800138000。")
        add(engine, "p9-h-2", "我喜欢跑步。")
        rows = engine.search(SearchRequest(query="我喜欢什么运动？", user_id="p9-user", top_k=5))
        assert rows
        assert "跑步" in rows[0]["content"]
    finally:
        if old is None:
            os.environ.pop("AML_USE_LLM", None)
        else:
            os.environ["AML_USE_LLM"] = old
