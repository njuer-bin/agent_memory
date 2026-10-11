"""P9 non-B2 improvements for AML.

This layer deliberately leaves B2 causal-chain reconstruction untouched. It
adds deterministic read-time coverage for the remaining weak dimensions:
current-state facts, temporal/event/rule recall, long-history compression,
experience retrieval, and privacy-aware ranking.
"""

from __future__ import annotations

import re
from typing import Any

from .models import new_id


_CURRENT_MARKERS = (
    "现在", "目前", "当前", "最新", "如今", "现居",
    "current", "currently", "right now", "at present", "latest", "now",
)
_SUMMARY_MARKERS = (
    "总结", "概括", "摘要", "整体", "总体", "全部", "所有", "一路以来",
    "记得什么", "关于我的", "我的经历", "我的历史",
    "summarize", "summary", "overall", "in general", "what do you remember",
    "what have you learned", "everything you know", "whole history",
)
_RULE_MARKERS = (
    "习惯", "通常", "一般", "规则", "原则", "偏好规则", "总是", "应该",
    "habit", "habits", "usually", "rule", "rules", "routine", "preference", "always",
)
_EXPERIENCE_MARKERS = (
    "经验", "教训", "以前怎么", "过去怎么", "类似情况", "上次", "曾经遇到",
    "学到了", "学会了", "怎么处理", "怎么解决", "应该怎么做",
    "experience", "lesson", "learned", "previously", "similar situation",
    "what should i do", "how did i handle", "how did i solve", "what worked",
)
_SENSITIVE_MARKERS = (
    "身份证", "证件", "护照", "银行卡", "银行账号", "账号", "密码", "验证码",
    "手机号", "手机号码", "电话号码", "邮箱", "电子邮箱", "住址", "详细地址",
    "身份证号", "社保号", "病历", "诊断", "精神", "心理", "医疗",
    "passport", "bank account", "password", "verification code", "phone number",
    "email address", "home address", "medical", "diagnosis", "mental health",
    "social security",
)


def _contains_any(query: str, markers: tuple[str, ...]) -> bool:
    q = query.lower()
    return any(marker.lower() in q for marker in markers)


def _lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query.lower()))
    t_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / max(1, len(q_tokens))


def _candidate(row: dict[str, Any], *, source: str, score: float, memory_type: str, metadata: dict | None = None) -> dict:
    item = {
        "id": str(row.get("id") or new_id("p9")),
        "content": str(row.get("content") or ""),
        "role": row.get("role", "memory"),
        "timestamp": int(row.get("timestamp") or row.get("valid_from") or 0),
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id", ""),
        "score": float(score),
        "source": source,
        "memory_type": memory_type,
        "status": row.get("status", "active"),
        "valid_from": row.get("valid_from", row.get("timestamp", 0)),
        "valid_to": row.get("valid_to"),
        "metadata": dict(metadata or {}),
    }
    return item


def _is_b2_query(query: str, plan: Any) -> bool:
    if getattr(plan, "multi_hop", False):
        lower = query.lower()
        return any(x in lower for x in (
            "为什么", "为何", "原因", "导致", "因为", "所以", "因此", "中间", "经过", "路径",
            "why", "cause", "caused", "because", "therefore", "led to", "resulted in", "path",
        ))
    return False


def _current_fact_candidates(engine, request, plan) -> list[dict]:
    if request.include_history or not _contains_any(request.query, _CURRENT_MARKERS):
        return []
    predicate = getattr(plan, "predicate_hint", None)
    if not predicate:
        return []
    out = []
    for fact in engine.store.active_facts(request.user_id, include_history=False):
        if str(fact.get("predicate") or "") != predicate:
            continue
        out.append(_candidate(
            fact,
            source="p9_current_state",
            score=1.06,
            memory_type="fact",
            metadata={
                "subject": fact.get("subject"),
                "predicate": fact.get("predicate"),
                "object": fact.get("object"),
                "p9_channel": "current_state",
                "p9_read_time": True,
            },
        ))
    return out[:8]


def _temporal_candidates(engine, request, plan) -> list[dict]:
    if not getattr(plan, "temporal", False):
        return []
    start = getattr(plan, "temporal_start", None)
    end = getattr(plan, "temporal_end", None)
    out = []
    for fact in engine.store.active_facts(request.user_id, include_history=True):
        ts = int(fact.get("valid_from") or fact.get("timestamp") or 0)
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        out.append(_candidate(fact, source="p9_temporal_fact", score=0.96,
                              memory_type="fact", metadata={"p9_channel": "temporal"}))
    for event in engine.store.events(request.user_id, start_time=start, end_time=end):
        out.append(_candidate(event, source="p9_temporal_event", score=0.94,
                              memory_type="event", metadata={"p9_channel": "temporal"}))
    out.sort(key=lambda x: (float(x.get("score", 0.0)), int(x.get("timestamp", 0))), reverse=True)
    return out[:12]


def _rule_candidates(engine, request) -> list[dict]:
    if not _contains_any(request.query, _RULE_MARKERS):
        return []
    rules = engine.store.rules(request.user_id)
    scored = []
    for rule in rules:
        overlap = _lexical_overlap(request.query, str(rule.get("content") or "") + " " + str(rule.get("rule") or ""))
        scored.append(_candidate(
            rule,
            source="p9_rule",
            score=0.90 + min(0.10, overlap * 0.10),
            memory_type="rule",
            metadata={"rule": rule.get("rule"), "p9_channel": "rule"},
        ))
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return scored[:10]


def _experience_candidates(engine, request) -> list[dict]:
    if not _contains_any(request.query, _EXPERIENCE_MARKERS):
        return []
    out = []
    for event in engine.store.events(request.user_id):
        text = str(event.get("content") or "") + " " + str(event.get("event") or "")
        overlap = _lexical_overlap(request.query, text)
        out.append(_candidate(event, source="p9_experience_event",
                              score=0.78 + min(0.12, overlap * 0.12),
                              memory_type="event", metadata={"p9_channel": "experience"}))
    for raw in engine.store.all_raw(request.user_id):
        content = str(raw.get("content") or "")
        overlap = _lexical_overlap(request.query, content)
        if overlap <= 0 and len(out) >= 20:
            continue
        out.append(_candidate(raw, source="p9_experience_raw",
                              score=0.72 + min(0.10, overlap * 0.10),
                              memory_type="raw", metadata={"p9_channel": "experience"}))
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:16]


def _summary_candidate(engine, request) -> dict | None:
    if not _contains_any(request.query, _SUMMARY_MARKERS):
        return None
    parts: list[str] = []
    for fact in engine.store.active_facts(request.user_id, include_history=True)[:10]:
        parts.append(str(fact.get("content") or "").strip())
    for event in engine.store.events(request.user_id)[:10]:
        parts.append(str(event.get("content") or "").strip())
    for rule in engine.store.rules(request.user_id)[:8]:
        parts.append(str(rule.get("content") or "").strip())
    for profile in engine.store.profiles(request.user_id)[:8]:
        parts.append(str(profile.get("content") or "").strip())
    unique = []
    seen = set()
    for part in parts:
        part = " ".join(part.split())
        if part and part not in seen:
            seen.add(part)
            unique.append(part)
    if not unique:
        return None
    return _candidate(
        {"id": f"p9-summary:{request.user_id}", "content": "历史记忆摘要：" + "\n".join(unique[:28]),
         "timestamp": max((int(x.get("timestamp") or 0) for x in engine.store.all_raw(request.user_id)), default=0)},
        source="p9_summary",
        score=1.01,
        memory_type="summary",
        metadata={"p9_channel": "summary", "p9_compression": True},
    )


def _apply_privacy(rows: list[dict], query: str) -> list[dict]:
    explicit = _contains_any(query, _SENSITIVE_MARKERS)
    if explicit:
        return rows
    for item in rows:
        content = str(item.get("content") or "")
        if _contains_any(content, _SENSITIVE_MARKERS):
            item["score"] = min(float(item.get("score", 0.0)), 0.02)
            md = dict(item.get("metadata") or {})
            md["p9_privacy"] = "deprioritized_unrequested_sensitive_memory"
            item["metadata"] = md
    return rows


def install_p9() -> None:
    from . import engine as engine_module

    if getattr(engine_module.MemoryEngine, "_p9_non_b2_installed", False):
        return
    original_search = engine_module.MemoryEngine.search

    def search_with_p9(self, request):
        rows = list(original_search(self, request))
        query = (request.query or request.question or "").strip()
        if not query:
            return rows
        try:
            latest = self.store.all_raw(request.user_id)
            reference_ts = latest[0]["timestamp"] if latest else None
            plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
            if _is_b2_query(query, plan):
                return rows

            by_id = {str(row.get("id")) for row in rows}
            additions = []
            additions.extend(_current_fact_candidates(self, request, plan))
            additions.extend(_temporal_candidates(self, request, plan))
            additions.extend(_rule_candidates(self, request))
            additions.extend(_experience_candidates(self, request))
            summary = _summary_candidate(self, request)
            if summary:
                additions.append(summary)

            for item in additions:
                if str(item.get("id")) not in by_id:
                    rows.append(item)
                    by_id.add(str(item.get("id")))

            rows = _apply_privacy(rows, query)
            rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return rows[: request.top_k]
        except Exception:
            return rows[: request.top_k]

    engine_module.MemoryEngine.search = search_with_p9
    engine_module.MemoryEngine._p9_non_b2_installed = True


install_p9()
