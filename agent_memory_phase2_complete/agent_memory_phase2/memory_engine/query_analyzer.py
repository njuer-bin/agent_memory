from __future__ import annotations

import re
from dataclasses import dataclass

from .temporal_parser import normalize_temporal


@dataclass
class QueryPlan:
    original: str
    rewritten: str
    multi_hop: bool
    keywords: list[str]
    temporal: bool
    temporal_start: int | None = None
    temporal_end: int | None = None
    temporal_relation: str = "at"
    memory_type_hint: str | None = None
    relation_hint: bool = False


class QueryAnalyzer:
    MULTI_HOP_MARKERS = (
        "谁推荐", "谁介绍", "朋友的", "同事的", "他的", "她的",
        "他们", "那个", "之前提到", "基于", "根据", "为什么",
        "和谁", "关系", "哪个朋友", "朋友推荐", "同事推荐",
    )

    TEMPORAL_MARKERS = (
        "现在", "目前", "当前", "以前", "之前", "后来", "之后",
        "最近", "当时", "历史", "过去", "去年", "前年", "今年",
        "曾经", "上个月", "本月", "昨天", "今天", "明天",
    )

    def analyze(self, query: str, forced_multi_hop=None, reference_ts=None) -> QueryPlan:
        q = query.strip()
        rewritten = self.rewrite(q)
        multi = any(x in q for x in self.MULTI_HOP_MARKERS)
        if forced_multi_hop is not None:
            multi = forced_multi_hop
        temporal = any(x in q for x in self.TEMPORAL_MARKERS)
        if reference_ts is None:
            import time
            reference_ts = int(time.time() * 1000)
        info = normalize_temporal(q, reference_ts)
        keywords = self.keywords(rewritten)
        memory_type_hint = self.infer_memory_type(q)
        relation_hint = any(
            marker in q for marker in (
                "朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁"
            )
        )
        return QueryPlan(
            q, rewritten, multi, keywords, temporal,
            info.start, info.end, info.relation,
            memory_type_hint, relation_hint
        )

    @staticmethod
    def infer_memory_type(q: str) -> str | None:
        if any(x in q for x in ("习惯", "通常", "一般", "规则", "请记住", "总是")):
            return "rule"
        if any(x in q for x in ("什么时候", "何时", "哪天", "哪一年", "参加了什么", "发生了什么")):
            return "event"
        if any(x in q for x in ("朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁")):
            return "relation"
        if any(x in q for x in ("喜欢", "爱好", "偏好", "不喜欢")):
            return "fact"
        if any(x in q for x in ("住哪里", "住哪", "居住地", "住过")):
            return "fact"
        return None

    @staticmethod
    def rewrite(q: str) -> str:
        replacements = {
            "我现在住哪": "用户 当前 居住地",
            "我现在住哪里": "用户 当前 居住地",
            "我住哪里": "用户 当前 居住地",
            "我住哪": "用户 当前 居住地",
            "以前住哪里": "用户 历史 居住地",
            "之前住哪里": "用户 历史 居住地",
            "我喜欢什么": "用户 喜欢 偏好",
            "我的爱好": "用户 喜欢 偏好",
        }
        out = q
        for a, b in replacements.items():
            out = out.replace(a, b)
        return out

    @staticmethod
    def keywords(q: str) -> list[str]:
        chars = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", q.lower())
        grams = []
        for i in range(len(chars) - 1):
            if all("\u4e00" <= c <= "\u9fff" for c in chars[i:i + 2]):
                grams.append(chars[i] + chars[i + 1])
        return list(dict.fromkeys(chars + grams))
