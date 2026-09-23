from __future__ import annotations

from .models import fingerprint


class MemoryGovernance:
    """
    写入治理：
    - 相同 user/subject/predicate/object 去重
    - 同一 user/subject/predicate 的新值覆盖旧 active 状态
    - 历史不删除，通过 superseded 状态保存
    """

    def __init__(self, store):
        self.store = store

    def accept_fact(self, fact):
        same = self.store.find_same_fact(
            fact.user_id, fact.subject, fact.predicate, fact.object
        )
        if same:
            return False, "duplicate"

        current = self.store.find_current_fact(
            fact.user_id, fact.subject, fact.predicate
        )

        if current and current["object"] != fact.object:
            self.store.update_fact_status(
                current["id"], "superseded", fact.timestamp
            )
            fact.supersedes_id = current["id"]

        self.store.insert_fact(fact)
        return True, "inserted"
