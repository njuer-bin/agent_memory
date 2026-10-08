from memory_engine.models import AddMessage, AddRequest, SearchRequest
from memory_engine.reasoning_engine import ReasoningMemoryEngine
from memory_engine.temporal_parser import normalize_temporal


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p7-test-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_current_state_prefers_governed_new_value(tmp_path):
    engine = ReasoningMemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p7-state-1", "s1", "我现在住在北京。", 1704067200000)
    add(engine, "p7-state-2", "s2", "我现在住在上海。", 1704153600000)

    rows = engine.search(SearchRequest(
        query="我现在住在哪里？",
        user_id="p7-test-user",
        top_k=5,
    ))
    assert rows
    assert any("上海" in row["content"] for row in rows[:2]), rows
    assert not any("北京" in row["content"] for row in rows[:2]), rows


def test_causal_reasoning_promotes_cause_evidence(tmp_path):
    engine = ReasoningMemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p7-cause-1", "s1", "因为项目延期，所以发布推迟了。", 1704067200000)

    rows = engine.search(SearchRequest(
        query="为什么发布推迟了？",
        user_id="p7-test-user",
        top_k=5,
        multi_hop=True,
    ))
    contents = [row["content"] for row in rows]
    assert any("项目延期" in content and "发布推迟" in content for content in contents), contents


def test_temporal_parser_supports_english_month_and_relative_interval():
    reference = 1704067200000  # 2024-01-01 UTC
    info = normalize_temporal("July 2022", reference)
    assert info.start is not None and info.end is not None
    assert info.relation == "at"

    info = normalize_temporal("4 years ago", reference)
    assert info.start is not None and info.end is not None
    assert info.relation == "before"
