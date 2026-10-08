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
    expanded_query: str = ""
    predicate_hint: str | None = None
    intent_hint: str | None = None


class QueryAnalyzer:
    MULTI_HOP_MARKERS = (
        "谁推荐", "谁介绍", "朋友的", "同事的", "他的", "她的", "他们",
        "那个", "之前提到", "基于", "根据", "为什么", "和谁", "关系",
        "哪个朋友", "朋友推荐", "同事推荐", "朋友工作", "朋友所在",
        "朋友公司", "朋友推荐的", "同事工作", "同事所在", "同事公司",
        "同事推荐的", "老板公司", "总部", "导致", "原因", "因为",
        "所以", "因此", "中间", "经过", "路径", "如何导致",
        "属于哪个国家", "属于什么国家", "总部所在城市", "所在城市属于", "也叫", "又名",
        "属于哪个国家", "属于什么国家", "总部所在城市", "所在城市属于", "也叫", "又名",
    )

    MULTI_HOP_PATTERNS = (
        re.compile(r"(?:朋友|同事|老板).{0,12}(?:工作|公司|所在|推荐|介绍|总部).{0,12}(?:哪|什么|哪里|谁|哪个|总部|城市|地方)"),
        re.compile(r"(?:他的|她的|他们的).{0,12}(?:公司|工作|朋友|同事|住处|城市|总部)"),
        re.compile(r"(?:推荐|介绍).{0,12}(?:的|给我|给用户).{0,12}(?:城市|地方|公司|人|对象)"),
        re.compile(r"(?:为什么|原因|导致|因为|所以|因此).{0,20}(?:什么|为何|为什么|结果|影响|导致)"),
        re.compile(r"(?:总部|公司).{0,12}(?:在哪里|在哪|什么地方|哪个城市)"),
        re.compile(r"(?:城市|地点).{0,12}(?:属于哪个国家|属于什么国家|哪个国家)"),
        re.compile(r"(?:也叫|又名|别名|简称).{0,20}(?:总部|公司|在哪里|在哪)"),
        re.compile(r"(?:城市|地点).{0,12}(?:属于哪个国家|属于什么国家|哪个国家)"),
        re.compile(r"(?:也叫|又名|别名|简称).{0,20}(?:总部|公司|在哪里|在哪)"),
    )

    TEMPORAL_MARKERS = (
        "现在", "目前", "当前", "以前", "之前", "后来", "之后",
        "最近", "当时", "历史", "过去", "去年", "前年", "今年",
        "曾经", "上个月", "本月", "上周", "本周", "下周", "昨天", "今天", "明天",
    )

    def analyze(self, query: str, forced_multi_hop=None, reference_ts=None) -> QueryPlan:
        q = query.strip()
        rewritten = self.rewrite(q)
        multi = (
            any(x in q for x in self.MULTI_HOP_MARKERS)
            or any(pattern.search(q) for pattern in self.MULTI_HOP_PATTERNS)
        )
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
                "朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁",
                "公司", "总部", "工作", "因果", "原因", "导致"
            )
        )
        expanded_query = self.expand_query(rewritten, memory_type_hint, relation_hint, info.relation)
        predicate_hint = self.infer_predicate(q)
        intent_hint = self.infer_intent(q, memory_type_hint)
        return QueryPlan(
            q, rewritten, multi, keywords, temporal,
            info.start, info.end, info.relation,
            memory_type_hint, relation_hint, expanded_query, predicate_hint, intent_hint
        )

    @staticmethod
    def infer_memory_type(q: str) -> str | None:
        if any(x in q for x in ("习惯", "通常", "一般", "规则", "请记住", "总是")):
            return "rule"
        if any(x in q for x in ("什么时候", "何时", "哪天", "哪一年", "参加了什么", "发生了什么")):
            return "event"
        if any(x in q for x in ("朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁", "公司", "总部", "工作", "导致", "原因")):
            return "relation"
        if any(x in q for x in ("喜欢", "爱好", "偏好", "不喜欢")):
            return "fact"
        if any(x in q for x in (
            "住哪里", "住哪", "居住地", "住过", "哪个国家", "什么国家",
            "哪个城市", "什么城市", "在哪里", "在哪", "属于哪里", "属于哪个国家"
        )):
            return "fact"
        return None

    @staticmethod
    def infer_predicate(q: str) -> str | None:
        if any(x in q for x in ("不喜欢", "讨厌", "不爱")):
            return "dislike"
        if any(x in q for x in ("喜欢", "偏好", "爱好", "喜爱")):
            return "like"
        if any(x in q for x in ("职业", "工作", "从事")):
            return "occupation"
        if any(x in q for x in ("住哪里", "住哪", "居住地", "住过", "住址")):
            return "residence"
        if "生日" in q or "出生" in q:
            return "birthday"
        if any(x in q for x in ("名字", "姓名", "叫")):
            return "name"
        return None

    @staticmethod
    def infer_intent(q: str, memory_type_hint: str | None) -> str | None:
        if memory_type_hint == "rule":
            return "habit"
        if memory_type_hint == "event":
            return "event"
        if memory_type_hint == "relation":
            return "relation"
        if memory_type_hint == "fact":
            return "fact"
        return None

    @staticmethod
    def expand_query(q: str, memory_type_hint: str | None, relation_hint: bool,
                      temporal_relation: str) -> str:
        terms = []
        if memory_type_hint == "rule":
            terms += ["习惯", "通常", "一般", "经常", "平时", "规则"]
        elif memory_type_hint == "fact":
            if any(x in q for x in ("偏好", "喜欢", "爱好", "喜爱", "不喜欢")):
                terms += ["偏好", "喜欢", "爱好", "喜爱"]
            if any(x in q for x in ("职业", "工作", "从事")):
                terms += ["职业", "工作", "从事"]
            if any(x in q for x in (
                "住哪里", "住哪", "居住地", "住过", "哪个国家", "什么国家",
                "哪个城市", "什么城市", "在哪里", "在哪", "属于哪里"
            )):
                terms += ["居住地", "住处", "居住", "城市", "国家", "地点", "以前", "曾经", "之前"]
        elif memory_type_hint == "event":
            terms += ["事件", "参加", "发生", "经历"]
        if relation_hint:
            terms += ["朋友", "好友", "同事", "推荐", "介绍", "关系", "公司", "总部", "工作"]
        if temporal_relation == "before":
            terms += ["以前", "之前", "曾经", "历史"]
        elif temporal_relation == "after":
            terms += ["后来", "之后"]
        unique = list(dict.fromkeys(x for x in terms if x not in q))
        return q if not unique else q + " " + " ".join(unique)

    @staticmethod
    def _english_evidence_queries(q: str) -> list[str]:
        """Build evidence-oriented subqueries for English benchmark questions."""
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|\d{4}", q)
        if not tokens:
            return []
        stop = {
            "what", "which", "who", "where", "when", "why", "how", "many",
            "much", "does", "did", "do", "has", "have", "had", "is", "are",
            "was", "were", "be", "been", "being", "the", "a", "an", "and",
            "or", "to", "of", "for", "in", "on", "at", "with", "from", "by",
            "that", "this", "these", "those", "they", "them", "their", "both",
            "some", "any", "all", "ever", "also", "really", "just", "till",
            "date", "currently", "mentioned",
        }
        entities = []
        for raw in tokens:
            if raw[0].isupper() and raw.lower() not in stop and raw not in entities:
                entities.append(raw)
        terms = [t.lower() for t in tokens if t.lower() not in stop and len(t) > 2]
        years = [t for t in tokens if re.fullmatch(r"\d{4}", t)]
        variants = []
        salient = terms[:5]
        for entity in entities[:4]:
            tail = [t for t in salient if t.lower() != entity.lower()][:4]
            variants.append(" ".join([entity, *tail]))
        if len(entities) >= 2:
            variants.append(" ".join(entities[:2]) + " common shared both")
        if years:
            for entity in entities[:2]:
                variants.append(" ".join([entity, *years[:2]]))
        return list(dict.fromkeys(x.strip() for x in variants if x.strip()))[:7]

    @staticmethod
    def evidence_requirements(plan: QueryPlan) -> list[dict]:
        """Plan independent evidence requirements for completeness-first retrieval."""
        q = plan.original.strip()
        lower = q.lower()
        requirements = []

        def add(kind, query, entity=None, temporal=None, priority=0):
            query = " ".join(str(query or "").split()).strip()
            if not query or any(r["query"].lower() == query.lower() for r in requirements):
                return
            requirements.append({
                "id": f"R{len(requirements) + 1}",
                "kind": kind, "query": query, "entity": entity,
                "temporal": temporal, "priority": priority,
            })

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", q)
        stop = {
            "what","which","who","where","when","why","how","many","much","does",
            "did","do","has","have","had","is","are","was","were","be","been",
            "the","a","an","and","or","to","of","for","in","on","at","with","from",
            "by","that","this","these","those","both","some","any","all","ever",
            "also","really","just","till","date","mentioned","participate",
        }
        entities = []
        for token in tokens:
            if token[0].isupper() and token.lower() not in stop and token not in entities:
                entities.append(token)

        years = re.findall(r"\b(?:19|20)\d{2}\b", q)
        months = re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            q, flags=re.I,
        )
        temporal = " ".join(dict.fromkeys([*months, *years]))
        if plan.temporal and not temporal:
            temporal = plan.original

        # Per-entity retrieval is mandatory for multi-entity questions.
        for entity in entities[:3]:
            add("entity", entity, entity=entity, priority=10)

        if len(entities) >= 2:
            pair = " ".join(entities[:2])
            if any(x in lower for x in ("both", "common", "shared", "same", "together")):
                add("intersection", f"{pair} common shared both", entity=pair, priority=20)
            else:
                add("pair", pair, entity=pair, priority=12)

        if temporal:
            add("temporal", " ".join([*entities[:2], temporal]), entity=entities[0] if entities else None,
                temporal=temporal, priority=22)

        # Preserve the full question for predicate-specific wording.
        add("full", plan.rewritten, priority=5)

        if not entities:
            chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", q)
            chinese_stop = {
                "什么","哪些","哪个","哪里","怎么","为什么","多少","有没有","是否",
                "以及","还有","关于","分别","他们","她们","这个","那个","之前","之后",
            }
            for term in [x for x in chinese_terms if x not in chinese_stop][:3]:
                add("term", term, entity=term, priority=10)

        # Causal questions need independent evidence slots for cause,
        # intermediate step, and outcome.  Treat these as evidence obligations
        # rather than merely adding causal words to the same full-query search.
        if any(x in q for x in ("为什么","原因","导致","因为","所以","因此","为何","how did","why")):
            add("causal_cause", q + " 原因 前因 为什么 因为", priority=30)
            add("causal_path", q + " 中间步骤 过程 导致 因此 所以", priority=29)
            add("causal_effect", q + " 结果 后果 影响 导致", priority=28)

        requirements.sort(key=lambda r: (-r["priority"], r["id"]))
        for idx, item in enumerate(requirements[:4], 1):
            item["id"] = f"R{idx}"
        return requirements[:4]

    @staticmethod
    def retrieval_queries(plan: QueryPlan) -> list[str]:
        """Generate bounded full-query and evidence-requirement retrieval views."""
        q = plan.rewritten
        variants = [q]
        if plan.multi_hop:
            if "朋友" in plan.original or "同事" in plan.original or "老板" in plan.original:
                variants += ["朋友 同事 老板 关系", "朋友 是谁", "朋友 推荐 工作 公司"]
            if any(x in plan.original for x in ("公司", "工作", "总部")):
                variants += ["工作 公司 就职", "公司 总部 位于", "总部 城市 地点"]
            if any(x in plan.original for x in ("推荐", "介绍")):
                variants += ["推荐 介绍", "推荐的 人 地方 公司"]
            if any(x in plan.original for x in ("属于哪个国家", "属于什么国家", "哪个国家")):
                variants += ["城市 国家 属于", "地点 国家", "属于 国家"]
            if any(x in plan.original for x in ("也叫", "又名", "别名", "简称")):
                variants += ["别名 也叫 又名", "alias canonical 总部 公司"]
            if any(x in plan.original for x in ("为什么", "原因", "导致", "因为", "所以", "因此")):
                variants += ["原因 因为 导致", "原因 结果 影响"]
            variants += QueryAnalyzer._english_evidence_queries(plan.original)
        if plan.expanded_query:
            variants.append(plan.expanded_query)
        return list(dict.fromkeys(x.strip() for x in variants if x.strip()))[:12]

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
            "我的偏好是什么": "用户 喜欢 偏好",
            "我的偏好是": "用户 喜欢 偏好",
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
