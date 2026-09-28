from __future__ import annotations

import re

from .models import fingerprint


def _norm(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s，,。；;、]+", "", value)
    return value


class MemoryGovernance:
    """写入治理：去重、冲突检测、superseded 链、纠正语义。"""

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

        # 语义归一化后仍相同，避免“北京”/“ 北京 ”之类重复。
        if current and _norm(current["object"]) == _norm(fact.object):
            return False, "semantic_duplicate"

        if current and current["object"] != fact.object:
            # 新事实从当前事实时间开始生效；旧事实保留为历史证据。
            self.store.update_fact_status(
                current["id"], "superseded", fact.timestamp
            )
            fact.supersedes_id = current["id"]

        self.store.insert_fact(fact)
        return True, "superseded" if current else "inserted"

    def accept_relation(self, relation):
        if self.store.find_same_relation(
            relation.user_id, relation.subject,
            relation.predicate, relation.object
        ):
            return False, "duplicate"
        self.store.insert_relation(relation)
        return True, "inserted"

    def accept_rule(self, rule):
        if self.store.find_same_rule(rule.user_id, rule.rule):
            return False, "duplicate"
        self.store.insert_rule(rule)
        return True, "inserted"
