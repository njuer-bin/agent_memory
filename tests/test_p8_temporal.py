from memory_engine.models import AddMessage, AddRequest, SearchRequest
from memory_engine.engine import MemoryEngine
from memory_engine.temporal_parser import normalize_temporal


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p8-temporal-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_relative_chinese_time_is_anchored():
    reference = 1704067200000  # 2024-01-01 UTC
    info = normalize_temporal("两天前", reference)
    assert info.start == reference - 2 * 86_400_000
    assert info.end == reference - 86_400_000
    assert info.relation == "before"

    info = normalize_temporal("上周", reference)
    assert info.start is not None and info.end is not None
    assert info.end >= info.start
    assert info.granularity == "week"


def test_historical_query_uses_anchored_interval(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p8-time-1", "s1", "我当时住在杭州。", 1672531200000)  # 2023-01-01
    add(engine, "p8-time-2", "s2", "我后来搬到了上海。", 1704067200000)  # 2024-01-01

    rows = engine.search(SearchRequest(
        query="去年我住在哪里？",
        user_id="p8-temporal-user",
        top_k=5,
    ))
    assert rows
    contents = [row["content"] for row in rows]
    assert any("杭州" in content for content in contents[:3]), contents


def test_temporal_bounds_are_preserved_for_explicit_request(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p8-time-3", "s1", "我在北京参加了会议。", 1704067200000)
    add(engine, "p8-time-4", "s2", "我在上海参加了会议。", 1735689600000)

    rows = engine.search(SearchRequest(
        query="参加会议",
        user_id="p8-temporal-user",
        top_k=5,
        start_time=1704067200000,
        end_time=1704067200000,
    ))
    contents = [row["content"] for row in rows]
    assert any("北京" in content for content in contents), contents
    assert not any("上海" in content for content in contents), contents
