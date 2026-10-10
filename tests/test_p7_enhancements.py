from memory_engine.analyzer import Fact
from memory_engine.p7_enhancements import P7MemoryAnalyzer, P7MemoryGovernance, P7QueryAnalyzer


class _FakeStore:
    def __init__(self):
        self.inserted = []
        self.conflicts = []

    def find_current_fact(self, user_id, subject, predicate):
        return {
            "id": "fact-current",
            "timestamp": 2000,
            "valid_from": 2000,
            "object": "Shanghai",
        }

    def insert_fact(self, fact):
        self.inserted.append(fact)

    def insert_conflict_log(self, row):
        self.conflicts.append(row)


def test_p7_english_current_state_query_is_typed():
    plan = P7QueryAnalyzer().analyze("Where do I currently live?", reference_ts=2_000_000)
    assert plan.predicate_hint == "residence"
    assert plan.temporal is True
    assert plan.memory_type_hint == "fact"


def test_p7_english_causal_relation_is_extracted():
    analyzer = P7MemoryAnalyzer()
    analyzed = analyzer.analyze(
        "u1",
        "The project was delayed because the supplier failed.",
        1_000,
    )
    causal = [r for r in analyzed["relations"] if r.predicate == "causes"]
    assert causal
    assert causal[0].subject == "the supplier failed"
    assert causal[0].object == "the project was delayed"


def test_p7_late_historical_fact_does_not_replace_current_state():
    store = _FakeStore()
    governance = P7MemoryGovernance(store)
    fact = Fact(
        id="fact-old",
        user_id="u1",
        subject="user",
        predicate="residence",
        object="Beijing",
        content="I lived in Beijing.",
        timestamp=1000,
        fingerprint="fp",
        valid_from=1000,
    )
    inserted, reason = governance.accept_fact(fact)
    assert inserted is True
    assert reason == "preserved_historical_fact"
    assert store.inserted == [fact]
    assert store.conflicts[0]["resolution"] == "preserved_historical_fact"
