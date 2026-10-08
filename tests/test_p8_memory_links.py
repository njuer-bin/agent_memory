from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p8-causal-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_memory_links_reconstruct_adjacent_causal_path(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "link-1", "s1", "数据库迁移没有完成。", 1704067200000)
    add(engine, "link-2", "s1", "因为数据库迁移没有完成，所以上线被推迟。", 1704067300000)
    add(engine, "link-3", "s1", "上线被推迟，后来客户投诉增加。", 1704067400000)

    links = engine.store.memory_links("p8-causal-user")
    assert links, links
    assert any(x["source_id"] == links[0]["source_id"] for x in links)

    rows = engine.search(SearchRequest(
        query="为什么客户投诉增加？",
        user_id="p8-causal-user",
        top_k=20,
    ))
    contents = [row["content"] for row in rows]
    assert any("数据库迁移没有完成" in x for x in contents), contents
    assert any("上线被推迟" in x for x in contents), contents
    assert any("客户投诉增加" in x for x in contents), contents


def test_link_builder_requires_strong_continuation_signal():
    from memory_engine.memory_links import MemoryLinkBuilder
    previous = [{
        "id": "a", "session_id": "s", "content": "我在北京工作", "timestamp": 1
    }]
    current = {
        "id": "b", "session_id": "s", "content": "北京项目后来确认有效", "timestamp": 2
    }
    links = MemoryLinkBuilder.build(current, previous)
    assert any(x["relation"] == "session_continuation" for x in links), links
