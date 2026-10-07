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


def test_zero_hops_disables_cross_memory_chain():
    builder = EvidenceChainBuilder(max_hops=0)
    candidates = [
        _item("m1", "我的朋友是 Bob", 0.80),
        _item("m2", "Bob 在 Acme 工作", 0.55),
    ]

    ranked = builder.annotate("朋友工作的公司", candidates)

    assert all(
        item["metadata"]["evidence_chain"] is False
        for item in ranked
    )


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


def test_conversational_variants_build_same_friend_bridge():
    builder = EvidenceChainBuilder(max_hops=2)
    variants = [
        "我的朋友叫 Bob",
        "我朋友 Bob",
        "Bob 是我的朋友",
        "我的朋友是 Bob！",
    ]
    for idx, text in enumerate(variants):
        ranked = builder.annotate(
            "朋友工作的公司",
            [
                _item(f"friend-{idx}", text, 0.6),
                _item(f"job-{idx}", "Bob 在 Acme 工作", 0.5),
            ],
        )
        by_id = {item["id"]: item for item in ranked}
        assert by_id[f"friend-{idx}"]["metadata"]["evidence_chain"] is True
        assert by_id[f"job-{idx}"]["metadata"]["evidence_chain"] is True


def test_multi_fact_message_is_split_into_independent_clauses():
    builder = EvidenceChainBuilder(max_hops=2)
    candidates = [
        _item("m1", "我朋友 Bob 在 Acme 工作，他之前住在杭州。", 0.8),
        _item("m2", "Acme 总部在上海。", 0.4),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)
    by_id = {item["id"]: item for item in ranked}

    assert by_id["m1"]["metadata"]["evidence_chain"] is True
    assert by_id["m2"]["metadata"]["evidence_chain"] is True


def test_unrelated_conversational_memory_stays_out_of_chain():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        _item("m1", "我朋友 Bob", 0.45),
        _item("m2", "Bob 在 Acme 工作", 0.40),
        _item("m3", "Acme 总部在上海", 0.35),
        _item("noise", "今天吃了火锅，晚上准备看电影。", 0.99),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)
    by_id = {item["id"]: item for item in ranked}

    assert all(
        by_id[mid]["metadata"]["evidence_chain"] is True
        for mid in ("m1", "m2", "m3")
    )
    assert by_id["noise"]["metadata"]["evidence_chain"] is False


def test_query_analyzer_detects_natural_multi_hop_without_force_flag():
    from memory_engine.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    queries = [
        "朋友工作的公司总部在哪里？",
        "我朋友所在的公司在哪个城市？",
        "同事推荐的地方叫什么？",
    ]

    for query in queries:
        plan = analyzer.analyze(query, forced_multi_hop=None)
        assert plan.multi_hop is True, query


def test_query_analyzer_does_not_turn_simple_relation_lookup_into_multi_hop():
    from memory_engine.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    simple_queries = [
        "我的朋友是谁？",
        "我有几个朋友？",
        "谁是我的同事？",
    ]

    for query in simple_queries:
        plan = analyzer.analyze(query, forced_multi_hop=None)
        assert plan.multi_hop is False, query


def test_evidence_chain_handles_mixed_facts_without_crossing_unrelated_component():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        _item("a1", "我的朋友 Bob", 0.60),
        _item("a2", "Bob 在 Acme 工作", 0.50),
        _item("a3", "Acme 总部在上海", 0.40),
        _item("b1", "我的同事 Carol", 0.39),
        _item("b2", "Carol 在 Beta 工作", 0.38),
        _item("b3", "Beta 总部在深圳", 0.37),
        _item("noise", "今天下雨了，晚上早点休息。", 0.99),
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)
    by_id = {item["id"]: item for item in ranked}

    assert all(by_id[mid]["metadata"]["evidence_chain"] for mid in ("a1", "a2", "a3"))
    assert all(by_id[mid]["metadata"]["evidence_chain"] for mid in ("b1", "b2", "b3"))
    assert by_id["noise"]["metadata"]["evidence_chain"] is False


def test_evidence_chain_keeps_structured_edges_when_text_is_conversational():
    builder = EvidenceChainBuilder(max_hops=3)
    candidates = [
        {
            **_item("s1", "嗯，我朋友就是 Bob。", 0.60),
            "metadata": {"subject": "user", "object": "Bob"},
        },
        {
            **_item("s2", "他现在在 Acme 上班。", 0.50),
            "metadata": {"subject": "Bob", "object": "Acme"},
        },
        {
            **_item("s3", "那家公司总部在上海。", 0.40),
            "metadata": {"subject": "Acme", "object": "上海"},
        },
    ]

    ranked = builder.annotate("朋友工作的公司总部在哪里？", candidates)
    by_id = {item["id"]: item for item in ranked}
    assert all(by_id[mid]["metadata"]["evidence_chain"] for mid in ("s1", "s2", "s3"))
