from memory_engine.evidence_chain import EvidenceChainBuilder


def _item(mid, content, score):
    return {
        "id": mid,
        "content": content,
        "score": score,
        "memory_type": "raw",
        "metadata": {},
    }


def test_evidence_chain_preserves_three_hop_bridge():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        _item("m1", "我的朋友是 Bob", 0.80),
        _item("m2", "Bob 在 Acme 工作", 0.55),
        _item("m3", "Acme 总部在上海", 0.20),
        _item("m4", "Carol 在 Other 工作", 0.70),
    ]

    ranked = builder.annotate("我的朋友工作的公司总部在哪里？", candidates)

    by_id = {item["id"]: item for item in ranked}
    assert by_id["m1"]["metadata"]["evidence_chain"] is True
    assert by_id["m2"]["metadata"]["evidence_chain"] is True
    assert by_id["m3"]["metadata"]["evidence_chain"] is True
    assert by_id["m3"]["score"] > 0.20
    assert by_id["m4"]["metadata"]["evidence_chain"] is False


def test_structured_relation_edges_are_supported():
    builder = EvidenceChainBuilder(max_hops=2)
    candidates = [
        {
            "id": "r1",
            "content": "用户的朋友是 Bob",
            "score": 0.8,
            "metadata": {"subject": "user", "object": "Bob"},
        },
        {
            "id": "r2",
            "content": "Bob 在 Acme 工作",
            "score": 0.2,
            "metadata": {"subject": "Bob", "object": "Acme"},
        },
    ]

    ranked = builder.annotate("朋友工作的公司", candidates)
    assert all(item["metadata"]["evidence_chain"] for item in ranked)
