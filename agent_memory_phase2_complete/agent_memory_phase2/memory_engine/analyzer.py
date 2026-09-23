from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import fingerprint, new_id


@dataclass
class Fact:
    id: str
    user_id: str
    subject: str
    predicate: str
    object: str
    content: str
    timestamp: int
    fingerprint: str
    valid_from: int
    valid_to: Optional[int] = None
    status: str = "active"


@dataclass
class Relation:
    id: str
    user_id: str
    subject: str
    predicate: str
    object: str
    content: str
    timestamp: int
    fingerprint: str


@dataclass
class Event:
    id: str
    user_id: str
    event: str
    content: str
    timestamp: int
    fingerprint: str


@dataclass
class Rule:
    id: str
    user_id: str
    rule: str
    content: str
    timestamp: int
    fingerprint: str


@dataclass
class Profile:
    user_id: str
    key: str
    value: str
    content: str
    timestamp: int


class MemoryAnalyzer:
    """
    轻量、确定性的 Memory Analyzer。

    设计目标不是替代大模型抽取，而是：
    1. 无外部模型也能运行；
    2. 对常见 Persona / LoCoMo 风格事实进行结构化抽取；
    3. 后续可以无缝替换成 LLM / 本地模型抽取器。
    """

    # 关系抽取规则：
    # 每个元素都是：
    #     (compiled_regex, predicate)
    REL_PATTERNS = [
        (
            re.compile(
                r"(?:我的|我)?"
                r"(?:朋友|同事|老板|妻子|丈夫|妈妈|母亲|爸爸|父亲|"
                r"姐姐|哥哥|弟弟|妹妹)"
                r"\s*([^\n，。,.]{1,20})"
            ),
            "related_person",
        ),
    ]

    # 事实抽取规则：
    # 每个元素都是：
    #     (compiled_regex, predicate)
    FACT_PATTERNS = [
        (
            re.compile(r"我(?:现在)?住在([^\n，。,.；;]{1,40})"),
            "residence",
        ),
        (
            re.compile(r"我(?:目前)?在([^\n，。,.；;]{1,40})(?:工作|上班)"),
            "workplace",
        ),
        (
            re.compile(r"我(?:的)?职业是([^\n，。,.；;]{1,40})"),
            "occupation",
        ),
        (
            re.compile(r"我喜欢([^\n。；;]{1,60})"),
            "like",
        ),
        (
            re.compile(r"我不喜欢([^\n。；;]{1,60})"),
            "dislike",
        ),
        (
            re.compile(r"我讨厌([^\n。；;]{1,60})"),
            "dislike",
        ),
        (
            re.compile(r"我最喜欢([^\n。；;]{1,60})"),
            "favorite",
        ),
        (
            re.compile(r"我的生日是([^\n，。,.；;]{1,30})"),
            "birthday",
        ),
        (
            re.compile(r"我的名字是([^\n，。,.；;]{1,30})"),
            "name",
        ),
        (
            re.compile(r"我叫([^\n，。,.；;]{1,30})"),
            "name",
        ),
        (
            re.compile(r"我常用的语言是([^\n，。,.；;]{1,30})"),
            "language",
        ),
    ]

    # 规则抽取：
    RULE_PATTERNS = [
        re.compile(r"(?:以后|今后|从现在开始)[，,:： ]*(.*)"),
        re.compile(r"(?:请记住|记住)[，,:： ]*(.*)"),
        re.compile(r"(?:我的习惯是)[，,:： ]*(.*)"),
    ]

    # 事件关键词
    EVENT_WORDS = (
        "搬到",
        "搬家",
        "毕业",
        "入职",
        "离职",
        "结婚",
        "分手",
        "旅行",
        "去过",
        "参加",
        "开始",
        "结束",
        "购买",
        "买了",
        "完成",
    )

    def analyze(
        self,
        user_id: str,
        content: str,
        timestamp: int,
    ):
        facts: list[Fact] = []
        relations: list[Relation] = []
        events: list[Event] = []
        rules: list[Rule] = []
        profiles: list[Profile] = []

        # ---------------------------------------------------------
        # 1. Fact / Profile
        # ---------------------------------------------------------
        for pattern, predicate in self.FACT_PATTERNS:
            for m in pattern.finditer(content):
                obj = m.group(1).strip(" ，,。；;")

                if not obj:
                    continue

                fp = fingerprint(
                    user_id,
                    "fact",
                    m.group(0),
                    predicate,
                    obj,
                )

                fact = Fact(
                    id=new_id("fact"),
                    user_id=user_id,
                    subject="user",
                    predicate=predicate,
                    object=obj,
                    content=m.group(0).strip(),
                    timestamp=timestamp,
                    fingerprint=fp,
                    valid_from=timestamp,
                )

                facts.append(fact)

                profiles.append(
                    Profile(
                        user_id=user_id,
                        key=predicate,
                        value=obj,
                        content=m.group(0).strip(),
                        timestamp=timestamp,
                    )
                )

        # ---------------------------------------------------------
        # 2. Relation
        # ---------------------------------------------------------
        #
        # REL_PATTERNS 的元素结构是：
        #
        #     (pattern, predicate)
        #
        # 所以这里必须进行 tuple unpacking。
        #
        for pattern, predicate in self.REL_PATTERNS:
            for m in pattern.finditer(content):
                value = m.group(1).strip()

                if not value:
                    continue

                relations.append(
                    Relation(
                        id=new_id("rel"),
                        user_id=user_id,
                        subject="user",
                        predicate=predicate,
                        object=value,
                        content=m.group(0).strip(),
                        timestamp=timestamp,
                        fingerprint=fingerprint(
                            user_id,
                            "rel",
                            "user",
                            predicate,
                            value,
                        ),
                    )
                )

        # ---------------------------------------------------------
        # 3. Rule
        # ---------------------------------------------------------
        for pattern in self.RULE_PATTERNS:
            match = pattern.search(content)

            if not match:
                continue

            value = match.group(1).strip()

            if not value:
                continue

            rules.append(
                Rule(
                    id=new_id("rule"),
                    user_id=user_id,
                    rule=value,
                    content=match.group(0).strip(),
                    timestamp=timestamp,
                    fingerprint=fingerprint(
                        user_id,
                        "rule",
                        value,
                    ),
                )
            )

        # ---------------------------------------------------------
        # 4. Event
        # ---------------------------------------------------------
        if any(word in content for word in self.EVENT_WORDS):
            event_content = content.strip()

            events.append(
                Event(
                    id=new_id("event"),
                    user_id=user_id,
                    event=self._event_name(content),
                    content=event_content,
                    timestamp=timestamp,
                    fingerprint=fingerprint(
                        user_id,
                        "event",
                        event_content,
                        timestamp,
                    ),
                )
            )

        return {
            "facts": facts,
            "relations": relations,
            "events": events,
            "rules": rules,
            "profiles": profiles,
        }

    @staticmethod
    def _event_name(content: str) -> str:
        for word in MemoryAnalyzer.EVENT_WORDS:
            if word in content:
                return word

        return "event"