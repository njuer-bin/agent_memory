from __future__ import annotations

"""P13: F1-only long-history compression.

P10 remains the canonical retrieval pipeline. This module only changes the
returned evidence for broad summary/history queries. Ordinary queries return
P10 results byte-for-byte (same objects and ordering).

The F1 path uses deterministic, loss-aware compression rather than a single
large dossier: facts, events, preferences/rules, and a chronological change
view are kept as separate evidence units. This gives the answer model several
compact, independently retrievable views of a user's long history while
preserving the underlying raw memory as the source of truth.
"""

import re
from collections import OrderedDict
from typing import Any


_SUMMARY_MARKERS = (
    "总结", "概括", "摘要", "整体", "总体", "全部", "所有", "历史",
    "以前", "之前", "过去", "长期", "记得什么", "记得我", "关于我的记忆",
    "关于我的", "我的经历", "我的历史", "我的变化", "这些年来", "一路以来",
    "summarize", "summary", "summarize my", "recap", "overall", "in general",
    "what do you remember", "what have you learned", "everything you know",
    "whole history", "my history", "my past", "over time", "across our conversations",
)


def is_f1_query(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    if any(marker in q for marker in _SUMMARY_MARKERS):
        return True
    return bool(
        re.search(r"\b(?:summarize|summary|recap|remember)\b.*\b(?:me|my|our)\b", q)
        or re.search(r"\b(?:everything|all)\b.*\b(?:about|you know)\b", q)
    )


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _clean(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _item(content: str, *, user_id: str, timestamp: int, view: str, ids: list[str], score: float) -> dict[str, Any]:
    return {
        "id": f"p13-f1:{user_id}:{view}",
        "content": content,
        "role": "context",
        "timestamp": timestamp,
        "user_id": user_id,
        "session_id": "__f1_summary__",
        "score": score,
        "source": "p13_f1_compression",
        "memory_type": "summary",
        "status": "active",
        "valid_from": 0,
        "valid_to": None,
        "metadata": {
            "p13_channel": "f1",
            "compression": True,
            "loss_aware": True,
            "view": view,
            "source_message_ids": ids,
        },
    }


def build_f1_views(engine, request) -> list[dict[str, Any]]:
    """Create several compact, complementary summary evidence units.

    No raw message is deleted or rewritten in storage. The views are read-time
    projections designed specifically for the platform's long-history answer
    generation. Each view has a different coverage objective so a single
    summary cannot become an information bottleneck.
    """
    user_id = request.user_id
    facts = list(engine.store.active_facts(user_id, include_history=True))
    events = list(engine.store.events(user_id))
    rules = list(engine.store.rules(user_id))
    profiles = list(engine.store.profiles(user_id))
    raws = list(engine.store.all_raw(user_id))

    # Keep source ordering deterministic. Facts/events may be newest-first;
    # chronological view explicitly reverses them later.
    facts.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    events.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    rules.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    profiles.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    raws.sort(key=lambda x: (int(x.get("timestamp") or 0), str(x.get("id") or "")), reverse=True)

    views: list[dict[str, Any]] = []

    fact_lines = _dedupe(
        [str(x.get("content") or "") for x in facts] + [str(x.get("content") or "") for x in profiles],
        28,
    )
    fact_ids = [str(x.get("id")) for x in facts + profiles if x.get("id")][:40]
    if fact_lines:
        views.append(_item(
            "长期记忆摘要｜事实与个人背景\n" + "\n".join(f"- {x}" for x in fact_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or x.get("valid_from") or 0) for x in facts + profiles] or [0]),
            view="facts_profile",
            ids=fact_ids,
            score=1.12,
        ))

    event_lines = _dedupe([str(x.get("content") or "") for x in events], 24)
    event_ids = [str(x.get("id")) for x in events if x.get("id")][:40]
    if event_lines:
        views.append(_item(
            "长期记忆摘要｜经历与事件\n" + "\n".join(f"- {x}" for x in event_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or x.get("valid_from") or 0) for x in events] or [0]),
            view="events",
            ids=event_ids,
            score=1.11,
        ))

    rule_lines = _dedupe(
        [str(x.get("content") or "") for x in rules]
        + [str(x.get("rule") or "") for x in rules],
        20,
    )
    rule_ids = [str(x.get("id")) for x in rules if x.get("id")][:40]
    if rule_lines:
        views.append(_item(
            "长期记忆摘要｜偏好、习惯与规则\n" + "\n".join(f"- {x}" for x in rule_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or x.get("valid_from") or 0) for x in rules] or [0]),
            view="rules_preferences",
            ids=rule_ids,
            score=1.10,
        ))

    # A dedicated timeline/change view prevents the classic summary failure:
    # knowing the current state but losing how it changed over time.
    timeline_source = facts + events
    timeline_source.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")))
    timeline_lines = _dedupe([str(x.get("content") or "") for x in timeline_source], 28)
    timeline_ids = [str(x.get("id")) for x in timeline_source if x.get("id")][:48]
    if timeline_lines:
        views.append(_item(
            "长期记忆摘要｜时间线与变化轨迹\n" + "\n".join(f"- {x}" for x in timeline_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or x.get("valid_from") or 0) for x in timeline_source] or [0]),
            view="timeline_changes",
            ids=timeline_ids,
            score=1.09,
        ))

    # If structured extraction is sparse, retain a compact raw-history view so
    # F1 does not fail simply because older conversations were not structured.
    if not views and raws:
        raw_lines = _dedupe([str(x.get("content") or "") for x in reversed(raws)], 32)
        raw_ids = [str(x.get("id")) for x in raws if x.get("id")][:48]
        views.append(_item(
            "长期记忆摘要｜原始历史补充\n" + "\n".join(f"- {x}" for x in raw_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or 0) for x in raws] or [0]),
            view="raw_history_fallback",
            ids=raw_ids,
            score=1.08,
        ))

    return views


def install_p13() -> None:
    from . import engine as engine_module

    if getattr(engine_module.MemoryEngine.search, "_p13_f1_installed", False):
        return

    original_search = engine_module.MemoryEngine.search

    def search_with_f1(self, request):
        query = (request.query or request.question or "").strip()
        # Hard isolation: every non-F1 query follows the P10 search path
        # without touching or re-ranking its returned rows.
        if not is_f1_query(query):
            return original_search(self, request)

        views = build_f1_views(self, request)
        if not views:
            return original_search(self, request)

        # F1 gets only the compact summary views. This intentionally does not
        # alter normal P10 retrieval, and prevents a summary blob from stealing
        # a slot from raw evidence on ordinary questions.
        return views[: max(1, int(request.top_k))]

    search_with_f1._p13_f1_installed = True
    engine_module.MemoryEngine.search = search_with_f1


install_p13()
