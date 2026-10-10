from types import SimpleNamespace

from memory_engine.p8_evidence import EvidenceCompletenessPool
from memory_engine.query_analyzer import QueryAnalyzer


class FakeHybrid:
    def candidates(self, **kwargs):
        q = kwargs["query"].lower()
        if "alice" in q:
            return [
                (
                    {
                        "id": "raw-alice",
                        "content": "Alice works at Acme.",
                        "metadata": {"entity": "Alice"},
                        "memory_type": "raw",
                        "status": "active",
                        "timestamp": 1,
                    },
                    0.8,
                )
            ]
        return []


class FakeStore:
    def relations(self, user_id):
        return [
            {
                "id": "rel-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "content": "Alice works at Acme.",
                "timestamp": 1,
                "status": "active",
            },
            {
                "id": "rel-2",
                "subject": "Acme",
                "predicate": "headquarters",
                "object": "Boston",
                "content": "Acme is headquartered in Boston.",
                "timestamp": 2,
                "status": "active",
            },
        ]


class FakeEngine:
    def __init__(self):
        self.hybrid = FakeHybrid()
        self.store = FakeStore()
        self.query_analyzer = QueryAnalyzer()


def make_request(query="Where is Alice's company headquartered?", multi_hop=True):
    return SimpleNamespace(
        user_id="u1",
        query=query,
        question="",
        multi_hop=multi_hop,
        top_k=5,
        include_history=False,
        session_id=None,
        start_time=None,
        end_time=None,
        memory_types=None,
    )


def test_p8_requirement_channel_has_provenance():
    engine = FakeEngine()
    request = make_request()
    plan = engine.query_analyzer.analyze(request.query, True, 10)
    rows = EvidenceCompletenessPool(engine)._requirement_candidates(request, plan)

    assert rows
    assert rows[0]["metadata"]["p8_channel"] == "requirement"
    assert rows[0]["metadata"]["p8_requirement_id"].startswith("R")
    assert rows[0]["score"] == EvidenceCompletenessPool.REQUIREMENT_SCORE


def test_p8_targeted_channel_only_runs_for_temporal_state_queries():
    engine = FakeEngine()
    pool = EvidenceCompletenessPool(engine)

    current = make_request("Where do I live currently?", False)
    current_plan = engine.query_analyzer.analyze(current.query, False, 10)
    rows = pool._targeted_candidates(current, current_plan)
    assert rows == [] or all(r["metadata"]["p8_channel"] in {"state", "temporal"} for r in rows)

    plain = make_request("What is Alice?", False)
    plain_plan = engine.query_analyzer.analyze(plain.query, False, 10)
    assert pool._targeted_candidates(plain, plain_plan) == []


def test_p8_bounded_bridge_recovers_two_hop_path():
    engine = FakeEngine()
    request = make_request()
    plan = engine.query_analyzer.analyze(request.query, True, 10)
    pool = EvidenceCompletenessPool(engine)
    rows = pool._bridge_candidates(request, [{"metadata": {"entity": "Alice"}}], plan)

    ids = {row["id"] for row in rows}
    assert "rel-1" in ids
    assert "rel-2" in ids
    assert max(row["metadata"]["p8_hop"] for row in rows) <= pool.BRIDGE_HOPS


def test_p8_path_promotion_keeps_complementary_hops():
    rows = [
        {"id": "r1", "score": 0.90, "metadata": {"p8_channel": "bridge", "p8_hop": 1}},
        {"id": "r2", "score": 0.20, "metadata": {"p8_channel": "bridge", "p8_hop": 2}},
        {"id": "r3", "score": 0.95, "metadata": {}},
    ]
    promoted = EvidenceCompletenessPool.promote_complementary_paths(rows, 3, True)
    ids = [row["id"] for row in promoted]
    assert "r1" in ids
    assert "r2" in ids


def test_p8_bridge_is_disabled_for_single_hop_queries():
    engine = FakeEngine()
    request = make_request("Where does Alice live?", False)
    plan = engine.query_analyzer.analyze(request.query, False, 10)
    assert EvidenceCompletenessPool(engine)._bridge_candidates(
        request, [{"metadata": {"entity": "Alice"}}], plan
    ) == []
