import os

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest
from memory_engine.p13_f1 import build_f1_views, is_f1_query


def _add(engine, request_id, session_id, text, ts):
    engine.add(AddRequest(
        request_id=request_id,
        messages=[AddMessage(role="user", content=text, timestamp=ts)],
        user_id="p13-user",
        session_id=session_id,
    ))


def test_f1_query_detection_is_narrow():
    assert is_f1_query("请总结一下我过去的经历")
    assert is_f1_query("what do you remember about me?")
    assert not is_f1_query("我现在住在哪里？")
    assert not is_f1_query("2024年发生了什么？")


def test_f1_views_cover_multiple_memory_types(tmp_path):
    os.environ["AML_USE_LLM"] = "0"
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    _add(engine, "r1", "s1", "我的职业是软件工程师。", 1000)
    _add(engine, "r2", "s1", "2024年我参加了上海科技展。", 2000)
    _add(engine, "r3", "s2", "我喜欢安静的工作环境。", 3000)

    request = SearchRequest(query="请总结一下我过去的经历", user_id="p13-user", top_k=5)
    views = build_f1_views(engine, request)

    assert views
    assert all(item["memory_type"] == "summary" for item in views)
    assert {item["metadata"]["view"] for item in views} >= {"facts_profile", "events"}
    assert all(item["metadata"]["loss_aware"] for item in views)


def test_non_f1_search_keeps_p10_path(tmp_path):
    os.environ["AML_USE_LLM"] = "0"
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    _add(engine, "r1", "s1", "我的职业是软件工程师。", 1000)

    request = SearchRequest(query="我的职业是什么？", user_id="p13-user", top_k=5)
    rows = engine.search(request)

    assert rows
    assert not any(row.get("source") == "p13_f1_compression" for row in rows)
