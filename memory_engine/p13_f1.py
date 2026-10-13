from __future__ import annotations

"""P13: F1-only long-history compression.

P10 remains the canonical retrieval pipeline. P13.1 only narrows the router
used by P13 so temporal/history questions do not accidentally enter the F1
compression path.
"""

import re
from typing import Any

# Only explicit global-synthesis intent should enter F1. Do not treat generic
# temporal words such as "之前/以前/过去" as F1 by themselves.
_F1_CN_PATTERNS = (
    r"(?:总结|概括|摘要|归纳).*(?:我|我的).*(?:经历|历史|情况|记忆|变化|偏好|整体)",
    r"(?:整体|总体|全面).*(?:总结|概括|回顾).*(?:我|我的)",
    r"(?:我|我的).*(?:经历|历史|记忆|变化).*(?:总结|概括|回顾|整体)",
    r"(?:这些年来|一路以来).*(?:我|我的).*(?:总结|概括|经历|变化)",
    r"(?:你还记得|你记得).*(?:我|我的).*(?:什么|哪些)",
)

_F1_EN_PATTERNS = (
    r"\b(?:summarize|summary|recap|review)\b.*\b(?:me|my|our)\b",
    r"\b(?:what do you remember|what have you learned)\b.*\b(?:about me|about my)\b",
    r"\b(?:everything|all)\b.*\b(?:about me|about my|you know about me)\b",
    r"\b(?:my|our)\b.*\b(?:history|past|experiences|journey)\b.*\b(?:summarize|summary|recap|overall)\b",
)


def is_f1_query(query: str) -> bool:
    """Return True only for explicit long-history/global-summary intent."""
    q = str(query or "").strip().lower()
    if not q:
        return False
    return any(re.search(pattern, q) for pattern in _F1_CN_PATTERNS + _F1_EN_PATTERNS)


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
    user_id = request.user_id
    facts = list(engine.store.active_facts(user_id, include_history=True))
    events = list(engine.store.events(user_id))
    rules = list(engine.store.rules(user_id))
    profiles = list(engine.store.profiles(user_id))
    raws = list(engine.store.all_raw(user_id))

    facts.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    events.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("valid_from") or 0)), reverse=True)
    rules.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    profiles.sort(key=lambda x: (int(x.get("timestamp") or x.get("valid_from") or 0), str(x.get("id") or "")), reverse=True)
    raws.sort(key=lambda x: (int(x.get("timestamp") or 0), str(x.get("id") or "")), reverse=True)

    views: list[dict[str, Any]] = []

    # Structured extraction is preferred, but raw history remains a loss-aware
    # fallback for information that was not classified into a structured type.
    fact_lines = _dedupe(
        [str(x.get("content") or "") for x in facts]
        + [str(x.get("content") or "") for x in profiles]
        + [str(x.get("content") or "") for x in raws],
        28,
    )
    fact_ids = [str(x.get("id")) for x in facts + profiles + raws if x.get("id")][:48]
    if fact_lines:
        views.append(_item(
            "长期记忆摘要｜事实与个人背景\n" + "\n".join(f"- {x}" for x in fact_lines),
            user_id=user_id,
            timestamp=max([int(x.get("timestamp") or x.get("valid_from") or 0) for x in facts + profiles + raws] or [0]),
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

    if not views and raws:
        fallback_lines = _dedupe([str(x.get("content") or "") for x in reversed(raws)], 32)
        raw_ids = [str(x.get("id")) for x in raws if x.get("id")][:48]
        views.append(_item(
            "长期记忆摘要｜原始历史补充\n" + "\n".join(f"- {x}" for x in fallback_lines),
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
        if not is_f1_query(query):
            return original_search(self, request)

        views = build_f1_views(self, request)
        if not views:
            return original_search(self, request)

        return views[: max(1, int(request.top_k))]

    search_with_f1._p13_f1_installed = True
    engine_module.MemoryEngine.search = search_with_f1


install_p13()
