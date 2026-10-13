from types import SimpleNamespace

from memory_engine.p14_g2 import _application_evidence, _parse_if_then, _rule_score


class FakeStore:
    def __init__(self):
        self._rules = [
            {
                "id": "r1",
                "user_id": "u",
                "rule": "如果早上有会议，就不要安排早班飞机",
                "content": "如果早上有会议，就不要安排早班飞机",
                "timestamp": 100,
            },
            {
                "id": "r2",
                "user_id": "u",
                "rule": "如果周末有空，就去跑步",
                "content": "如果周末有空，就去跑步",
                "timestamp": 90,
            },
        ]

    def rules(self, user_id):
        return list(self._rules)


def test_parse_if_then():
    assert _parse_if_then("如果早上有会议，就不要安排早班飞机") == ("早上有会议", "不要安排早班飞机")


def test_rule_score_requires_real_overlap():
    score, overlap = _rule_score("早上有会议时应该怎么安排", "如果早上有会议，就不要安排早班飞机")
    assert score > 0.12
    assert overlap


def test_application_returns_grounded_rule():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(
        user_id="u",
        query="如果明天早上有会议，我应该怎么安排飞机？",
        question="",
        multi_hop=False,
        top_k=5,
    )
    rows = _application_evidence(engine, request)
    assert rows
    assert rows[0]["metadata"]["application"] is True
    assert "不要安排早班飞机" in rows[0]["content"]


def test_non_rule_question_does_not_activate_g2():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(
        user_id="u",
        query="我今天住在哪里？",
        question="",
        multi_hop=False,
        top_k=5,
    )
    assert _application_evidence(engine, request) == []
