import os
from types import SimpleNamespace

os.environ["AML_USE_LLM"] = "0"

from memory_engine.p11_non_b2 import _global_history_summary, _rule_application


class FakeStore:
    def __init__(self):
        self.raw = [
            {"id": "r1", "user_id": "u", "session_id": "s1", "timestamp": 100, "content": "我开始学习摄影"},
            {"id": "r2", "user_id": "u", "session_id": "s1", "timestamp": 200, "content": "后来我完成了第一组作品"},
            {"id": "r3", "user_id": "u", "session_id": "s2", "timestamp": 300, "content": "我换了工作并开始远程办公"},
            {"id": "r4", "user_id": "u", "session_id": "s3", "timestamp": 400, "content": "我准备参加一个摄影比赛"},
        ]
        self._rules = [
            {"id": "rule1", "user_id": "u", "rule": "重要任务先备份", "content": "遇到重要任务应该先备份资料", "timestamp": 350},
            {"id": "rule2", "user_id": "u", "rule": "早睡", "content": "通常晚上十一点前休息", "timestamp": 250},
        ]

    def all_raw(self, user_id, session_id=None):
        rows = [x for x in self.raw if x["user_id"] == user_id]
        if session_id is not None:
            rows = [x for x in rows if x["session_id"] == session_id]
        return sorted(rows, key=lambda x: x["timestamp"], reverse=True)

    def active_facts(self, user_id, include_history=True):
        return [{"id": "f1", "user_id": user_id, "content": "我偏好远程办公", "timestamp": 310, "predicate": "work_mode"}]

    def events(self, user_id, **kwargs):
        return [{"id": "e1", "user_id": user_id, "content": "参加摄影比赛", "timestamp": 410}]

    def rules(self, user_id):
        return list(self._rules)

    def profiles(self, user_id):
        return []


def test_global_summary_preserves_all_sessions_and_is_compressed():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(user_id="u", query="总结一下我的长期历史", include_history=True)
    rows = _global_history_summary(engine, request)
    assert rows
    global_row = rows[0]
    assert global_row["metadata"]["compressed"] is True
    assert global_row["metadata"]["session_count"] == 3
    assert "s1" in global_row["content"]
    assert "s2" in global_row["content"]
    assert "s3" in global_row["content"]
    assert "摄影比赛" in global_row["content"]


def test_rule_application_ranks_matching_rule_first():
    engine = SimpleNamespace(store=FakeStore())
    request = SimpleNamespace(user_id="u", query="遇到重要任务应该遵循什么规则")
    rows = _rule_application(engine, request)
    assert rows
    assert rows[0]["id"] == "rule1"
    assert rows[0]["metadata"]["applicable"] is True
    assert rows[0]["metadata"]["match_tokens"] > 0
