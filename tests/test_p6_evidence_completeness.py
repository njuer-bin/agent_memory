from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def add(engine, rid, session, content, ts):
    engine.add(AddRequest(
        request_id=rid,
        user_id="p6-test-user",
        session_id=session,
        messages=[AddMessage(role="user", content=content, timestamp=ts)],
    ))


def test_raw_message_kept_with_provenance(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p6-1", "s1", "我在 Acme 工作。", 1704067200000)

    rows = engine.hybrid.candidates("p6-test-user", "Acme 工作", top_k=100)
    raw = next(x for x, _ in rows if x["memory_type"] == "raw")
    assert raw["metadata"]["source_message_ids"] == [raw["id"]]


def test_same_session_neighbor_evidence_is_retrievable(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p6-1", "s1", "我最近认识了一个朋友小王。", 1704067200000)
    add(engine, "p6-2", "s1", "他现在在 Acme 工作。", 1704067300000)
    add(engine, "p6-3", "s1", "那家公司总部在上海。", 1704067400000)

    rows = engine.search(SearchRequest(
        query="我那个朋友工作的公司总部在哪里？",
        user_id="p6-test-user",
        top_k=100,
    ))
    contents = [x["content"] for x in rows]
    assert any("小王" in x and "Acme" in x for x in contents), contents
    assert any("Acme" in x and "上海" in x for x in contents), contents


def test_window_view_carries_all_source_message_ids(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p6-1", "s1", "第一条背景。", 1704067200000)
    add(engine, "p6-2", "s1", "第二条证据。", 1704067300000)
    add(engine, "p6-3", "s1", "第三条结论。", 1704067400000)

    rows = engine.hybrid.candidates("p6-test-user", "第二条证据", top_k=100)
    windows = [
        x for x, _ in rows
        if x["memory_type"] == "window" and "p6-test-user" == x["user_id"]
    ]
    assert windows
    assert any(
        len(x["metadata"].get("source_message_ids", [])) >= 2
        and "第二条证据。" in x["content"]
        for x in windows
    ), windows



def test_embedding_splits_oversized_input(monkeypatch):
    from memory_engine.vector import EmbeddingProvider

    provider = EmbeddingProvider()
    seen_lengths = []

    def fake_embed_one(text):
        seen_lengths.append(len(text))
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(provider, "_embed_one", fake_embed_one)
    vector = provider.embed("large-message-token " * 5000)

    import os
    max_chars = max(512, int(os.getenv("OLLAMA_EMBED_MAX_CHARS", "6000")))
    assert len(seen_lengths) > 1
    assert max(seen_lengths) <= max_chars
    assert len(vector) == 3
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-6


def test_neighbor_evidence_does_not_cross_sessions(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    add(engine, "p6-isolation-1", "s1", "我认识朋友小李。", 1704067200000)
    add(engine, "p6-isolation-2", "s2", "他在 Contoso 工作。", 1704067300000)

    rows = engine.search(SearchRequest(
        query="朋友工作的公司是什么？",
        user_id="p6-test-user",
        session_id="s1",
        top_k=100,
    ))
    contents = [x["content"] for x in rows]
    assert not any("小李" in x and "Contoso" in x for x in contents)
