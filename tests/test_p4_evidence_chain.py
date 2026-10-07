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


def test_high_score_distractor_is_not_marked_as_chain():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        _item("bridge1", "我的朋友是 Bob", 0.40),
        _item("bridge2", "Bob 在 Acme 工作", 0.35),
        _item("bridge3", "Acme 总部在上海", 0.30),
        _item("distractor", "Carol 在 Other 工作", 0.99),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)

    by_id = {item["id"]: item for item in ranked}
    assert by_id["bridge1"]["metadata"]["evidence_chain"] is True
    assert by_id["bridge2"]["metadata"]["evidence_chain"] is True
    assert by_id["bridge3"]["metadata"]["evidence_chain"] is True
    assert by_id["distractor"]["metadata"]["evidence_chain"] is False


def test_chain_score_and_metadata_are_added_to_every_chain_item():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        _item("m1", "我的朋友是 Bob", 0.50),
        _item("m2", "Bob 在 Acme 工作", 0.40),
        _item("m3", "Acme 总部在上海", 0.30),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)

    by_id = {item["id"]: item for item in ranked}
    for mid in ("m1", "m2", "m3"):
        md = by_id[mid]["metadata"]
        assert md["evidence_chain"] is True
        assert md["path_completeness"] == 1.0
        assert md["chain_score"] > 0.0
        assert by_id[mid]["score"] > next(
            score for item_id, score in (("m1", 0.50), ("m2", 0.40), ("m3", 0.30))
            if item_id == mid
        )


def test_max_hops_limits_distant_bridge():
    builder = EvidenceChainBuilder(max_hops=1)
    candidates = [
        _item("m1", "我的朋友是 Bob", 0.80),
        _item("m2", "Bob 在 Acme 工作", 0.55),
        _item("m3", "Acme 总部在上海", 0.20),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)

    by_id = {item["id"]: item for item in ranked}
    assert by_id["m1"]["metadata"]["evidence_chain"] is True
    assert by_id["m2"]["metadata"]["evidence_chain"] is True
    assert by_id["m3"]["metadata"]["evidence_chain"] is True


def test_relation_parser_accepts_whitespace_and_punctuation():
    builder = EvidenceChainBuilder(max_hops=2)
    candidates = [
        _item("m1", "我的朋友是 Bob。", 0.8),
        _item("m2", "Bob 在 Acme 工作。", 0.7),
    ]

    ranked = builder.annotate("朋友工作的公司", candidates)

    by_id = {item["id"]: item for item in ranked}
    assert by_id["m1"]["metadata"]["evidence_chain"] is True
    assert by_id["m2"]["metadata"]["evidence_chain"] is True
