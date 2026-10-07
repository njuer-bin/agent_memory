from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="b-test-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def search(engine, query, top_k=5):
    return engine.search(SearchRequest(
        query=query,
        user_id="b-test-user",
        top_k=top_k,
    ))


def test_cross_session_three_hop_evidence_chain(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-1", "s1", "我的朋友小王。", 1704067200000)
    add(engine, "b-2", "s2", "小王在 Acme 工作。", 1704067300000)
    add(engine, "b-3", "s3", "Acme 总部在上海。", 1704067400000)

    rows = search(engine, "我朋友工作的公司总部在哪里？", 5)
    contents = [row["content"] for row in rows]

    assert any("小王" in x for x in contents), contents
    assert any("Acme" in x for x in contents), contents
    assert any("上海" in x for x in contents), contents


def test_recommendation_chain(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-r1", "s1", "我的朋友叫小王。", 1704067200000)
    add(engine, "b-r2", "s2", "小王推荐我去杭州。", 1704067300000)

    rows = search(engine, "朋友推荐的城市是什么？", 5)
    assert any("杭州" in row["content"] for row in rows), rows


def test_causal_chain(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-c1", "s1", "项目延期。", 1704067200000)
    add(engine, "b-c2", "s2", "因为项目延期，所以上线推迟。", 1704067300000)

    rows = search(engine, "为什么上线推迟？", 5)
    contents = [row["content"] for row in rows]
    assert any("项目延期" in x and "上线推迟" in x for x in contents), contents


def test_controlled_open_domain_anchor(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-o1", "s1", "我现在住在上海。", 1704067200000)

    rows = search(engine, "我住的城市属于哪个国家？", 5)
    assert any("上海" in row["content"] for row in rows), rows


def test_reverse_causal_direction(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-c3", "s1", "上线推迟是因为项目延期。", 1704067200000)

    relations = engine.store.relations("b-test-user")
    causal = [
        r for r in relations
        if r.get("predicate") == "causes"
    ]
    assert any(
        r.get("subject") == "项目延期" and r.get("object") == "上线推迟"
        for r in causal
    ), causal


def test_alias_and_coreference_chain(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-a1", "s1", "Acme 也叫 艾克米。", 1704067200000)
    add(engine, "b-a2", "s2", "这个公司总部在上海。", 1704067300000)

    relations = engine.store.relations("b-test-user")
    assert any(
        r.get("predicate") == "alias_of"
        and r.get("subject") == "Acme"
        and r.get("object") == "艾克米"
        for r in relations
    ), relations
    assert any(
        r.get("predicate") == "headquarters"
        and r.get("subject") == "Acme"
        and r.get("object") == "上海"
        for r in relations
    ), relations
