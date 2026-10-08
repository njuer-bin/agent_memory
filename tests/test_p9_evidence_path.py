from memory_engine.evidence_path import EvidencePathReconstructor


def test_reconstructs_bounded_path_and_recovers_missing_node():
    reconstructor = EvidencePathReconstructor(max_hops=3, beam_width=4, neighbors_per_node=4)
    candidates = [
        {
            "id": "m1",
            "content": "最初决定参加活动",
            "score": 0.80,
            "_evidence_requirements": ["R1"],
            "metadata": {},
        },
        {
            "id": "m3",
            "content": "最终没有参加",
            "score": 0.82,
            "_evidence_requirements": ["R3"],
            "metadata": {},
        },
    ]
    links = [
        {"source_id": "m1", "target_id": "m2", "relation": "session_continuation", "confidence": 0.90},
        {"source_id": "m2", "target_id": "m3", "relation": "session_continuation", "confidence": 0.92},
    ]

    paths, promotions = reconstructor.reconstruct(
        query="为什么最终没有参加活动",
        candidates=candidates,
        requirement_plans=[
            {"id": "R1", "query": "最初决定", "priority": 10},
            {"id": "R3", "query": "最终没有参加", "priority": 10},
        ],
        link_rows=links,
        relation_rows=[],
    )

    assert paths
    assert any(path.depth <= 3 for path in paths)
    assert "m2" in promotions
    assert promotions["m2"]["path_depth"] >= 1


def test_path_search_is_bounded_and_prefers_requirement_coverage():
    reconstructor = EvidencePathReconstructor(max_hops=2, beam_width=2, neighbors_per_node=2)
    candidates = [
        {"id": "a", "content": "A", "score": 0.9, "_evidence_requirements": ["R1"], "metadata": {}},
        {"id": "b", "content": "B", "score": 0.8, "_evidence_requirements": ["R2"], "metadata": {}},
        {"id": "c", "content": "C", "score": 0.7, "_evidence_requirements": ["R3"], "metadata": {}},
    ]
    links = [
        {"source_id": "a", "target_id": "b", "relation": "topic_continuation", "confidence": 0.9},
        {"source_id": "b", "target_id": "c", "relation": "causal", "confidence": 0.95},
    ]

    paths, _ = reconstructor.reconstruct(
        query="multi hop",
        candidates=candidates,
        requirement_plans=[
            {"id": "R1", "query": "A"},
            {"id": "R2", "query": "B"},
            {"id": "R3", "query": "C"},
        ],
        link_rows=links,
        relation_rows=[],
    )

    assert paths
    assert max(path.depth for path in paths) <= 2
    assert max(len(path.covered_requirements) for path in paths) >= 2
