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
    source: str = "user"
    conflict_status: str = "none"
    conflict_group_id: Optional[str] = None


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
    """Deterministic memory parser with relation/causal extraction.

    Relation extraction is intentionally dependency-free.  It acts as the
    always-available fallback for hosted evaluation; an LLM extractor can be
    layered on later without changing the storage contract.
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

    # Ordered from specific conversational forms to generic forms.  The
    # extracted relation is persisted, so later search does not have to infer
    # the graph from raw text again.
    REL_PATTERNS = [
        # "我朋友小王推荐我去杭州" -> 小王 --recommends--> 杭州
        (re.compile(r"(?:我的|我|用户的)?(?:朋友|好友|同事|老板)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})\s*(推荐|介绍)(?:我去|我到|我|给我去|给我)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "recommend"),
        # "小王在 Acme 工作" / "小王就职于 Acme"
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:在|位于)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:工作|上班)"), "work_at"),
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:工作于|就职于)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "work_at"),
        # "Acme 总部在上海" / "Acme 位于上海"
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*总部\s*(?:在|位于|是)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "headquarters"),
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:位于|坐落于)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "located_in"),
        # Generic recommendation / introduction relations.
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(推荐|介绍)\s*(?:我|给我)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "recommend"),
        # "A 是 B 的朋友" / "A 属于 B"
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:是|叫|为)\s*(?:我的|我|用户的)?(朋友|同事|老板)"), "social_role"),
        (re.compile(r"([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:属于|隶属于|来自)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})"), "belongs_to"),
        # Existing user->person forms.
        (re.compile(r"(?:我的|我)?(?:朋友|好友)\s*(?:是|叫|为)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})(?=推荐|介绍|、|，|。|\s|$)"), "friend"),
        (re.compile(r"(?:我的|我)?同事\s*(?:是|叫|为)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})(?=推荐|介绍|、|，|。|\s|$)"), "colleague"),
        (re.compile(r"(?:我的|我)?老板\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "boss"),
        (re.compile(r"(?:我的|我)?(?:妈妈|母亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "mother"),
        (re.compile(r"(?:我的|我)?(?:爸爸|父亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "father"),
        (re.compile(r"(?:我的|我)?妻子\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "wife"),
        (re.compile(r"(?:我的|我)?丈夫\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "husband"),
    ]

    CAUSAL_PATTERNS = (
        re.compile(r"(?:因为|由于)\s*(.{1,60}?)\s*(?:，|,)?\s*(?:所以|因此|于是)\s*(.{1,60})"),
        re.compile(r"(.{1,40}?)\s*(?:导致|造成|引发|使得)\s*(.{1,40})"),
        re.compile(r"(.{1,40}?)\s*(?:是因为|源于)\s*(.{1,40})"),
    )

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

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip(" \t\r\n，。,.；;、:：!?！？“”\"'（）()")

    def _relation(self, user_id, subject, predicate, object_, content, timestamp):
        subject = self._clean(subject)
        object_ = self._clean(object_)
        if not subject or not object_ or subject == object_:
            return None
        if subject in {"我", "我的", "用户"}:
            subject = "user"
        if object_ in {"我", "我的", "用户"}:
            object_ = "user"
        return Relation(
            id=new_id("rel"),
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            object=object_,
            content=content.strip(),
            timestamp=timestamp,
            fingerprint=fingerprint(user_id, "rel", subject, predicate, object_),
        )

    def analyze(self, user_id: str, content: str, timestamp: int, source: str = "user"):
        facts: list[Fact] = []
        relations: list[Relation] = []
        events: list[Event] = []
        rules: list[Rule] = []
        profiles: list[Profile] = []

        temporal = normalize_temporal(content, timestamp)
        temporal_text = temporal.text

        for pattern, predicate in self.FACT_PATTERNS:
            for m in pattern.finditer(content):
                current_predicate = predicate
                obj = self._clean(m.group(1))
                if not obj:
                    continue
                if current_predicate == "like" and obj.startswith(("不", "讨厌")):
                    current_predicate = "dislike"

                fact_text = m.group(0).strip()
                fp = fingerprint(user_id, "fact", "user", current_predicate, obj)
                valid_from = temporal.start if temporal.start is not None else timestamp
                valid_to = temporal.end
                facts.append(Fact(
                    id=new_id("fact"), user_id=user_id, subject="user",
                    predicate=current_predicate, object=obj, content=fact_text,
                    timestamp=timestamp, fingerprint=fp, valid_from=valid_from,
                    valid_to=valid_to, temporal_text=temporal_text, source=source,
                ))
                profiles.append(Profile(
                    user_id=user_id, key=current_predicate, value=obj,
                    content=fact_text, timestamp=timestamp
                ))

        for pattern, predicate in self.REL_PATTERNS:
            for m in pattern.finditer(content):
                groups = m.groups()
                if len(groups) == 2:
                    subject, object_ = groups
                else:
                    subject, object_ = groups[0], groups[-1]
                rel = self._relation(user_id, subject, predicate, object_, m.group(0), timestamp)
                if rel:
                    relations.append(rel)

        # Causal edges are directed: cause -> effect.  The relation content
        # remains the original sentence so downstream evidence can cite it.
        for pattern in self.CAUSAL_PATTERNS:
            for m in pattern.finditer(content):
                left, right = self._clean(m.group(1)), self._clean(m.group(2))
                if left and right:
                    rel = self._relation(user_id, left, "causes", right, m.group(0), timestamp)
                    if rel:
                        relations.append(rel)

        # Deduplicate relation fingerprints inside a single message.
        unique_relations = []
        seen_fp = set()
        for rel in relations:
            if rel.fingerprint not in seen_fp:
                seen_fp.add(rel.fingerprint)
                unique_relations.append(rel)
        relations = unique_relations

        for pattern in self.RULE_PATTERNS:
            for match in pattern.finditer(content):
                value = match.group(1).strip(" ，,。；;")
                if value:
                    rules.append(Rule(
                        id=new_id("rule"), user_id=user_id, rule=value,
                        content=match.group(0).strip(), timestamp=timestamp,
                        fingerprint=fingerprint(user_id, "rule", value)
                    ))

        for word in self.EVENT_WORDS:
            if word in content:
                events.append(Event(
                    id=new_id("event"), user_id=user_id, event=word,
                    content=content.strip(), timestamp=timestamp,
                    fingerprint=fingerprint(user_id, "event", content.strip(), word),
                    event_start=temporal.start, event_end=temporal.end,
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
