from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryPlan:
    original: str
    rewritten: str
    multi_hop: bool
    keywords: list[str]
    temporal: bool


class QueryAnalyzer:
    MULTI_HOP_MARKERS = (
        "谁推荐", "谁介绍", "朋友的", "同事的", "他的", "她的",
        "他们", "那个", "之前提到", "基于", "根据", "为什么",
        "和谁", "关系", "哪个朋友", "朋友推荐", "同事推荐"
    )

    TEMPORAL_MARKERS = (
        "现在", "目前", "当前", "以前", "之前", "后来", "之后",
        "最近", "当时", "历史", "过去", "去年", "今年", "曾经"
    )

    def analyze(self, query: str, forced_multi_hop=None) -> QueryPlan:
        q = query.strip()
        rewritten = self.rewrite(q)
        multi = any(x in q for x in self.MULTI_HOP_MARKERS)
        if forced_multi_hop is not None:
            multi = forced_multi_hop
        temporal = any(x in q for x in self.TEMPORAL_MARKERS)
        keywords = self.keywords(rewritten)
        return QueryPlan(q, rewritten, multi, keywords, temporal)

    @staticmethod
    def rewrite(q: str) -> str:
        replacements = {
            "我现在住哪": "用户 当前 居住地",
            "我住哪里": "用户 当前 居住地",
            "我住哪": "用户 当前 居住地",
            "以前住哪里": "用户 历史 居住地",
            "之前住哪里": "用户 历史 居住地",
            "我喜欢什么": "用户 喜欢 偏好",
            "我的爱好": "用户 喜欢 偏好",
        }
        out = q
        for a,b in replacements.items():
            if a in out:
                out = out.replace(a,b)
        return out

    @staticmethod
    def keywords(q: str) -> list[str]:
        # 中文按单字 + 连续英文/数字抽取；保留常见两字词。
        chars = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", q.lower())
        grams = []
        for i in range(len(chars)-1):
            if all("\u4e00" <= c <= "\u9fff" for c in chars[i:i+2]):
                grams.append(chars[i] + chars[i+1])
        return list(dict.fromkeys(chars + grams))
