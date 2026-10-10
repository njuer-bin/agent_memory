from types import SimpleNamespace

from memory_engine.p8_evidence import EvidenceCompletenessPool
from memory_engine.query_analyzer import QueryAnalyzer


class FakeHybrid:
    def candidates(self, **kwargs):
        q = kwargs["query"]
        if "Alice" in q:
            return [(
                {
                    "id": "raw-alice",
                    "content": "Alice works at Acme.",
                    "metadata": {"entity": "Alice"},
                    "memory_type": "raw",
                },
                0.8,
            )]
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


def request():
    return SimpleNamespace(
        user_id="u1",
        query="Where is Alice's company headquartered?",
        question="",
        multi_hop=True,
        top_k=5,
        include_history=False,
        session_id=None,
        start_time=None,
        end_time=None,
        memory_types=None,
    )


def test_p8_requirement_channel_returns_provenance():
    engine = FakeEngine()
    plan = engine.query_analyzer.analyze(request().query, True, 10)
    pool = EvidenceCompletenessPool(engine)
    rows = pool._requirement_candidates(request(), plan)

    assert rows
    assert rows[0]["metadata"]["p8_channel"] == "requirement"
    assert rows[0]["metadata"]["p8_requirement_id"].startswith("R")
    assert rows[0]["score"] == pool.REQUIREMENT_SCORE


def test_p8_bounded_bridge_recovers_two_hop_path():
    engine = FakeEngine()
    plan = engine.query_analyzer.analyze(request().query, True, 10)
    pool = EvidenceCompletenessPool(engine)
    seed = [{"metadata": {"entity": "Alice"}}]

    rows = pool._bridge_candidates(request(), seed, plan)
    ids = {row["id"] for row in rows}

    assert "rel-1" in ids
    assert "rel-2" in ids
    assert max(row["metadata"]["p8_hop"] for row in rows) <= pool.BRIDGE_HOPS


def test_p8_is_fail_open_for_non_multi_hop():
    engine = FakeEngine()
    plan = engine.query_analyzer.analyze("Where does Alice live?", False, 10)
    pool = EvidenceCompletenessPool(engine)

    assert pool._bridge_candidates(request(), [{"metadata": {"entity": "Alice"}}], plan) == []
