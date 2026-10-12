"""P11: long-history compression/global synthesis and rule application.

P11 is layered on top of P10 and never mutates P10.  It keeps raw memories as
canonical evidence, then adds compact read-time views for two weak benchmark
areas: F1 long-history/global synthesis and G2 rule application.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .models import new_id

_SUMMARY = (
    "总结", "概括", "摘要", "整体", "总体", "全部", "所有", "一路以来",
    "我的经历", "我的历史", "记得什么", "关于我的", "长期", "过去一段时间",
    "summarize", "summary", "overall", "whole history", "what do you remember",
    "what have you learned", "long history", "in general",
)
_RULE = (
    "规则", "原则", "习惯", "通常", "一般", "应该", "总是", "如果", "遇到",
    "怎么处理", "怎么做", "偏好", "准则", "rule", "rules", "habit", "usually",
    "routine", "always", "should", "when", "how should", "guideline", "policy",
)
_STOP = {
    "我", "你", "他", "她", "它", "的", "了", "是", "在", "和", "与", "以及",
    "什么", "怎么", "如何", "哪些", "一下", "我的", "一下", "the", "a", "an", "what",
    "how", "do", "i", "you", "my", "is", "are", "and", "or", "to", "of",
}


def _has(text: str, markers: tuple[str, ...]) -> bool:
    q = (text or "").lower()
    return any(x.lower() in q for x in markers)


def _tokens(text: str) -> set[str]:
    return {
        x.lower() for x in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text or "")
        if x.lower() not in _STOP
    }


def _row(row: dict, score: float, source: str, memory_type: str, channel: str, content: str | None = None) -> dict:
    ts = int(row.get("timestamp") or row.get("valid_from") or row.get("event_start") or 0)
    return {
        "id": str(row.get("id") or new_id("p11")),
        "content": str(content if content is not None else row.get("content") or ""),
        "role": row.get("role", "memory"),
        "timestamp": ts,
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id", ""),
        "score": float(score),
        "source": source,
        "memory_type": memory_type,
        "status": row.get("status", "active"),
        "valid_from": row.get("valid_from", ts),
        "valid_to": row.get("valid_to"),
        "metadata": {"p11_channel": channel, "p11_read_time": True},
    }


def _compress_lines(rows: list[dict], limit: int = 4) -> list[str]:
    """Keep high-information unique lines instead of copying a whole session."""
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    for row in rows:
        text = " ".join(str(row.get("content") or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        # Structured memories carry more semantic density than conversational filler.
        memory_type = str(row.get("memory_type") or "")
        bonus = 3 if memory_type in {"fact", "event", "rule", "profile"} else 0
        scored.append((bonus, int(row.get("timestamp") or 0), text))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored[:limit]]


def _global_history_summary(engine, request) -> list[dict]:
    if not _has(request.query, _SUMMARY):
        return []

    user_id = request.user_id
    raw = engine.store.all_raw(user_id)
    sessions: dict[str, list[dict]] = defaultdict(list)
    for item in raw:
        sessions[str(item.get("session_id") or "__default__")].append({**item, "memory_type": "raw"})

    # Structured memories are folded into the same session buckets when their
    # source session is known; otherwise they remain global evidence.
    structured: list[dict] = []
    for name, getter in (("fact", "active_facts"), ("event", "events"), ("rule", "rules"), ("profile", "profiles")):
        fn = getattr(engine.store, getter, None)
        if not fn:
            continue
        try:
            rows = fn(user_id)
        except TypeError:
            rows = fn(user_id, include_history=True) if name == "fact" else []
        for row in rows or []:
            item = dict(row)
            item["memory_type"] = name
            structured.append(item)
            sid = str(item.get("session_id") or "")
            if sid and sid in sessions:
                sessions[sid].append(item)

    session_views: list[dict] = []
    for sid, rows in sessions.items():
        rows.sort(key=lambda r: int(r.get("timestamp") or r.get("valid_from") or r.get("event_start") or 0))
        lines = _compress_lines(rows, limit=4)
        if not lines:
            continue
        start = int(rows[0].get("timestamp") or rows[0].get("valid_from") or 0)
        end = int(rows[-1].get("timestamp") or rows[-1].get("valid_from") or 0)
        content = f"会话 {sid} 压缩摘要：\n" + "\n".join(f"- {x}" for x in lines)
        item = _row(
            {"id": f"p11-session:{user_id}:{sid}", "user_id": user_id, "session_id": sid, "timestamp": end},
            1.08,
            "p11_session_compression",
            "summary",
            "global_history",
            content,
        )
        item["metadata"].update({"session_start": start, "session_end": end, "compressed_items": len(lines)})
        session_views.append(item)

    session_views.sort(key=lambda x: int(x["timestamp"]), reverse=True)

    # Global synthesis is deliberately evidence-oriented: it compresses each
    # session but preserves every session's identity and representative facts.
    blocks = [x["content"] for x in session_views[:16]]
    if not blocks and structured:
        lines = _compress_lines(structured, limit=16)
        blocks = ["- " + x for x in lines]
    if not blocks:
        return []

    global_content = "全局长期历史综合（压缩视图）：\n" + "\n".join(blocks)
    global_row = _row(
        {"id": f"p11-global:{user_id}", "user_id": user_id, "timestamp": max(x["timestamp"] for x in session_views) if session_views else 0},
        1.32,
        "p11_global_synthesis",
        "summary",
        "global_history",
        global_content,
    )
    global_row["metadata"].update({
        "session_count": len(session_views),
        "compressed": True,
        "source_session_ids": [x.get("session_id", "") for x in session_views[:16]],
        "structured_memory_count": len(structured),
    })
    return [global_row] + session_views[:8]


def _rule_application(engine, request) -> list[dict]:
    if not _has(request.query, _RULE):
        return []
    fn = getattr(engine.store, "rules", None)
    if not fn:
        return []
    rules = fn(request.user_id) or []
    q_tokens = _tokens(request.query)
    out: list[dict] = []
    for rule in rules:
        text = " ".join(str(rule.get(k) or "") for k in ("rule", "content"))
        overlap = len(q_tokens & _tokens(text))
        # Rules are useful even with zero lexical overlap because the query is
        # explicitly asking for policy/behavior; overlap only determines order.
        score = 1.12 + min(0.28, overlap * 0.07)
        item = _row(rule, score, "p11_rule_application", "rule", "rule_application")
        item["metadata"].update({
            "rule": rule.get("rule"),
            "match_tokens": overlap,
            "application_query": request.query,
            "applicable": True,
        })
        out.append(item)
    out.sort(key=lambda x: (float(x["score"]), int(x.get("timestamp") or 0)), reverse=True)
    return out[:8]


def _install() -> None:
    from . import engine as engine_module
    if getattr(engine_module.MemoryEngine, "_p11_installed", False):
        return
    previous_search = engine_module.MemoryEngine.search

    def search_with_p11(self, request):
        rows = list(previous_search(self, request))
        query = (request.query or request.question or "").strip()
        if not query:
            return rows
        try:
            additions = _global_history_summary(self, request)
            additions.extend(_rule_application(self, request))
            seen = {str(x.get("id")) for x in rows}
            for item in additions:
                if str(item.get("id")) not in seen:
                    rows.append(item)
                    seen.add(str(item.get("id")))
            rows.sort(key=lambda x: (float(x.get("score", 0)), int(x.get("timestamp") or 0)), reverse=True)
            return rows[: request.top_k]
        except Exception:
            return rows[: request.top_k]

    engine_module.MemoryEngine.search = search_with_p11
    engine_module.MemoryEngine._p11_installed = True


_install()
