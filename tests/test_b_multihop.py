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


def test_four_hop_unordered_chain_with_distractors(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    # Intentionally add the bridge evidence out of chronological/session order.
    add(engine, "b-4", "s4", "上海属于中国。", 1704067500000)
    add(engine, "b-1", "s1", "我的朋友小王。", 1704067200000)
    add(engine, "b-x", "sx", "我的另一个朋友小李在北京工作。", 1704067250000)
    add(engine, "b-3", "s3", "Acme 总部在上海。", 1704067400000)
    add(engine, "b-2", "s2", "小王在 Acme 工作。", 1704067300000)

    rows = search(engine, "我朋友工作的公司总部所在城市属于哪个国家？", 8)
    contents = [row["content"] for row in rows]

    assert any("小王" in x and "Acme" in x for x in contents) or any("小王" in x for x in contents), contents
    assert any("Acme" in x and "上海" in x for x in contents), contents
    assert any("上海" in x and "中国" in x for x in contents), contents


def test_reverse_causal_multi_hop(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-c4", "s1", "上线推迟是因为项目延期。", 1704067200000)
    add(engine, "b-c5", "s2", "项目延期导致客户投诉。", 1704067300000)

    rows = search(engine, "为什么会有客户投诉？", 6)
    contents = [row["content"] for row in rows]
    assert any("项目延期导致客户投诉" in x for x in contents), contents
    assert any("上线推迟是因为项目延期" in x for x in contents), contents


def test_alias_query_can_reach_canonical_entity(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-al1", "s1", "Acme 也叫 艾克米。", 1704067200000)
    add(engine, "b-al2", "s2", "Acme 总部在上海。", 1704067300000)

    rows = search(engine, "艾克米总部在哪里？", 6)
    contents = [row["content"] for row in rows]
    assert any("上海" in x for x in contents), contents


def test_coreference_across_multiple_sessions(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-rf1", "s1", "我最近负责 Acme 项目。", 1704067200000)
    add(engine, "b-rf2", "s2", "这个项目导致上线延期。", 1704067300000)

    relations = engine.store.relations("b-test-user")
    assert any(
        r.get("predicate") == "causes"
        and r.get("subject") == "Acme"
        and "上线延期" in r.get("object", "")
        for r in relations
    ), relations


def test_relationship_attribution_survives_distractor(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "b-at1", "s1", "我的朋友小王推荐我去杭州。", 1704067200000)
    add(engine, "b-at2", "s2", "我的朋友小李推荐我去北京。", 1704067300000)
    add(engine, "b-at3", "s3", "小王在 Acme 工作。", 1704067400000)

    rows = search(engine, "小王工作的公司是什么？", 6)
    contents = [row["content"] for row in rows]
    assert any("小王" in x and "Acme" in x for x in contents), contents
