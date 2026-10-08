from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p8-causal-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_causal_chain_preserves_cause_intermediate_and_outcome(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "p8-1", "s1", "数据库迁移没有完成。", 1704067200000)
    add(engine, "p8-2", "s1", "因为数据库迁移没有完成，所以上线被推迟。", 1704067300000)
    add(engine, "p8-3", "s2", "上线被推迟导致客户投诉增加。", 1704067400000)

    rows = engine.search(SearchRequest(
        query="为什么客户投诉增加？",
        user_id="p8-causal-user",
        top_k=20,
    ))
    contents = [row["content"] for row in rows]

    assert any("数据库迁移没有完成" in x for x in contents), contents
    assert any("上线被推迟" in x for x in contents), contents
    assert any("客户投诉增加" in x for x in contents), contents


def test_causal_requirements_are_independently_planned(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    plan = engine.query_analyzer.analyze("为什么客户投诉增加？")
    reqs = engine.query_analyzer.evidence_requirements(plan)
    kinds = {r["kind"] for r in reqs}

    assert "causal_cause" in kinds, reqs
    assert "causal_path" in kinds, reqs
    assert "causal_effect" in kinds, reqs


def test_causal_relation_is_directed(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))

    add(engine, "p8-r1", "s1", "上线被推迟是因为数据库迁移没有完成。", 1704067200000)

    relations = engine.store.relations("p8-causal-user")
    assert any(
        r.get("predicate") == "causes"
        and r.get("subject") == "数据库迁移没有完成"
        and r.get("object") == "上线被推迟"
        for r in relations
    ), relations
