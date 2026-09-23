from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import fingerprint, new_id
from .temporal_parser import normalize_temporal


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
    supersedes_id: Optional[str] = None
    temporal_text: str = ""


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
    event_start: Optional[int] = None
    event_end: Optional[int] = None
    temporal_text: str = ""


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
    """Deterministic Parse-2 analyzer.

    覆盖：
    - atomic fact / profile
    - relation
    - rule / preference
    - event + temporal expression
    - 否定、纠正、迁移类事实
    - 一句话中的多事实抽取

    不依赖外部模型，后续可以把 analyze() 替换成 LLM extractor。
    """

    FACT_PATTERNS = [
        (re.compile(r"我(?:现在|目前|当前)?住在([^\n，。,.；;]+)"), "residence"),
        (re.compile(r"我(?:目前|现在)?在([^\n，。,.；;]+?)(?:工作|上班)"), "workplace"),
        (re.compile(r"我(?:的)?职业是([^\n，。,.；;]+)"), "occupation"),
        (re.compile(r"我最喜欢([^\n。；;，,]+)"), "favorite"),
        (re.compile(r"我喜欢([^\n。；;，,]+)"), "like"),
        (re.compile(r"我不喜欢([^\n。；;，,]+)"), "dislike"),
        (re.compile(r"我讨厌([^\n。；;，,]+)"), "dislike"),
        (re.compile(r"我的生日是([^\n，。,.；;]+)"), "birthday"),
        (re.compile(r"我的名字是([^\n，。,.；;]+)"), "name"),
        (re.compile(r"我叫([^\n，。,.；;]+)"), "name"),
        (re.compile(r"我常用的语言是([^\n，。,.；;]+)"), "language"),
        (re.compile(r"我来自([^\n，。,.；;]+)"), "origin"),
        (re.compile(r"我毕业于([^\n，。,.；;]+)"), "school"),
    ]

    REL_PATTERNS = [
        (re.compile(r"(?:我的|我)?(?:朋友|好友)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "friend"),
        (re.compile(r"(?:我的|我)?同事\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "colleague"),
        (re.compile(r"(?:我的|我)?老板\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "boss"),
        (re.compile(r"(?:我的|我)?(?:妈妈|母亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "mother"),
        (re.compile(r"(?:我的|我)?(?:爸爸|父亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "father"),
        (re.compile(r"(?:我的|我)?妻子\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "wife"),
        (re.compile(r"(?:我的|我)?丈夫\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "husband"),
    ]

    RULE_PATTERNS = [
        re.compile(r"(?:以后|今后|从现在开始)[，,:： ]*(.*)"),
        re.compile(r"(?:请记住|记住)[，,:： ]*(.*)"),
        re.compile(r"(?:我的习惯是)[，,:： ]*(.*)"),
        re.compile(r"(?:我通常|我一般)(.*)"),
    ]

    EVENT_WORDS = (
        "搬到", "搬家", "毕业", "入职", "离职", "结婚", "分手",
        "旅行", "去过", "参加", "开始", "结束", "购买", "买了",
        "完成", "搬去", "搬来", "加入", "辞职", "回到",
    )

    CORRECTION_MARKERS = ("不是", "改成", "改为", "其实是", "更正为", "纠正一下")

    def analyze(self, user_id: str, content: str, timestamp: int):
        facts: list[Fact] = []
        relations: list[Relation] = []
        events: list[Event] = []
        rules: list[Rule] = []
        profiles: list[Profile] = []

        temporal = normalize_temporal(content, timestamp)
        temporal_text = temporal.text

        for pattern, predicate in self.FACT_PATTERNS:
            for m in pattern.finditer(content):
                obj = m.group(1).strip(" ，,。；;")
                if not obj:
                    continue

                # “我不喜欢X”不应被 like 规则截断为“不喜欢X”
                if predicate == "like" and obj.startswith(("不", "讨厌")):
                    predicate = "dislike"

                fact_text = m.group(0).strip()
                fp = fingerprint(user_id, "fact", "user", predicate, obj)

                valid_from = temporal.start if temporal.start is not None else timestamp
                valid_to = temporal.end
                facts.append(Fact(
                    id=new_id("fact"),
                    user_id=user_id,
                    subject="user",
                    predicate=predicate,
                    object=obj,
                    content=fact_text,
                    timestamp=timestamp,
                    fingerprint=fp,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    temporal_text=temporal_text,
                ))
                profiles.append(Profile(
                    user_id=user_id, key=predicate, value=obj,
                    content=fact_text, timestamp=timestamp
                ))

        for pattern, predicate in self.REL_PATTERNS:
            for m in pattern.finditer(content):
                value = m.group(1).strip()
                if not value or value in {"推荐我", "介绍我", "告诉我", "说"}:
                    continue
                relations.append(Relation(
                    id=new_id("rel"),
                    user_id=user_id,
                    subject="user",
                    predicate=predicate,
                    object=value,
                    content=m.group(0).strip(),
                    timestamp=timestamp,
                    fingerprint=fingerprint(user_id, "rel", "user", predicate, value),
                ))

        for pattern in self.RULE_PATTERNS:
            for match in pattern.finditer(content):
                value = match.group(1).strip(" ，,。；;")
                if value:
                    rules.append(Rule(
                        id=new_id("rule"), user_id=user_id, rule=value,
                        content=match.group(0).strip(), timestamp=timestamp,
                        fingerprint=fingerprint(user_id, "rule", value)
                    ))

        # 事件：一条消息只要包含事件词就保留原句作为证据，并结构化时间。
        for word in self.EVENT_WORDS:
            if word in content:
                events.append(Event(
                    id=new_id("event"), user_id=user_id, event=word,
                    content=content.strip(), timestamp=timestamp,
                    fingerprint=fingerprint(user_id, "event", content.strip(), word),
                    event_start=temporal.start,
                    event_end=temporal.end,
                    temporal_text=temporal_text,
                ))
                break

        return {
            "facts": facts,
            "relations": relations,
            "events": events,
            "rules": rules,
            "profiles": profiles,
            "temporal": temporal,
            "correction": any(x in content for x in self.CORRECTION_MARKERS),
        }
