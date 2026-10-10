from __future__ import annotations

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest
from memory_engine.vector import EmbeddingProvider


def test_embedding_splits_oversized_input(monkeypatch):
    provider = EmbeddingProvider()
    seen_lengths = []

    def fake_embed_one(text):
        seen_lengths.append(len(text))
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(provider, "_embed_one", fake_embed_one)
    vector = provider.embed("large-message-token " * 5000)

    max_chars = max(512, int(__import__("os").getenv("OLLAMA_EMBED_MAX_CHARS", "6000")))
    assert len(seen_lengths) > 1
    assert max(seen_lengths) <= max_chars
    assert len(vector) == 3
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-6


def test_neighbor_evidence_stays_within_same_session(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    # Make this regression test deterministic and independent of a local Ollama service.
    engine.embedder.use_ollama = False
    user_id = "p6-test-user"

    messages = [
        ("p6-1", "我最近认识了一个朋友小王。", 1704067200000, "s1"),
        ("p6-2", "他现在在 Acme 工作。", 1704067300000, "s1"),
        ("p6-3", "那家公司总部在上海。", 1704067400000, "s1"),
        ("p6-4", "另一个会话的机密片段。", 1704067500000, "s2"),
    ]
    for request_id, content, timestamp, session_id in messages:
        engine.add(AddRequest(
            request_id=request_id,
            messages=[AddMessage(role="user", content=content, timestamp=timestamp)],
            user_id=user_id,
            session_id=session_id,
        ))

    rows = engine.search(SearchRequest(
        query="我那个朋友工作的公司总部在哪里？",
        user_id=user_id,
        top_k=100,
    ))
    contents = [row["content"] for row in rows]

    assert any("小王" in content and "Acme" in content for content in contents)
    assert not any(
        "另一个会话的机密片段" in content and "小王" in content
        for content in contents
    )


def test_neighbor_context_obeys_explicit_session_filter(tmp_path):
    engine = MemoryEngine(str(tmp_path / "memory.db"))
    engine.embedder.use_ollama = False
    user_id = "session-boundary-user"

    for request_id, content, timestamp, session_id in [
        ("s1-a", "我认识朋友小李。", 1704067200000, "s1"),
        ("s2-a", "他在 Contoso 工作。", 1704067300000, "s2"),
    ]:
        engine.add(AddRequest(
            request_id=request_id,
            messages=[AddMessage(role="user", content=content, timestamp=timestamp)],
            user_id=user_id,
            session_id=session_id,
        ))

    rows = engine.search(SearchRequest(
        query="朋友工作的公司是什么？",
        user_id=user_id,
        session_id="s1",
        top_k=100,
    ))
    contents = [row["content"] for row in rows]
    assert not any("小李" in content and "Contoso" in content for content in contents)
