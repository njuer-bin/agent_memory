from __future__ import annotations

"""P12: long-history retrieval without replacing canonical evidence.

The P11 experiment showed that summary-first retrieval can erase dates and
atomic facts. P12 therefore keeps raw memory canonical and adds auxiliary
summary/dossier evidence only for broad/global-history questions. The dossier
contains original messages with session and timestamp provenance; it is not a
lossy replacement for raw evidence.
"""

import re
from collections import defaultdict
from typing import Any


_GLOBAL_HISTORY_PATTERNS = (
    "历史", "以前", "之前", "过去", "长期", "整体", "总结", "总的来说",
    "全部", "所有", "记得什么", "记得我", "关于我的记忆", "我的经历",
    "我的变化", "这些年来", "一路以来", "what do you remember",
    "what you remember", "remember about me", "my history", "my past",
    "overall", "in general", "across our conversations", "over time",
)


def is_global_history_query(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    if any(marker in q for marker in _GLOBAL_HISTORY_PATTERNS):
        return True
    return bool(
        re.search(r"\b(?:summarize|summary|recap|remember)\b.*\b(?:me|our|my)\b", q)
        or re.search(r"\b(?:everything|all)\b.*\b(?:about|you know)\b", q)
    )


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", str(text or "").lower()))
    return {w for w in words if len(w) <= 32}


def _build_compat_summary(engine, user_id: str) -> dict[str, Any] | None:
    """Restore P9's summary candidate as an auxiliary compatibility channel.

    P12 intentionally does not use this summary as the canonical evidence.
    It exists so summary/history queries retain the P9 ``memory_type=summary``
    contract while the lossless raw dossier remains available alongside it.
    """
    parts: list[str] = []
    for fact in engine.store.active_facts(user_id, include_history=True)[:10]:
        parts.append(str(fact.get("content") or "").strip())
    for event in engine.store.events(user_id)[:10]:
        parts.append(str(event.get("content") or "").strip())
    for rule in engine.store.rules(user_id)[:8]:
        parts.append(str(rule.get("content") or "").strip())
    for profile in engine.store.profiles(user_id)[:8]:
        parts.append(str(profile.get("content") or "").strip())

    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = " ".join(part.split())
        if part and part not in seen:
            seen.add(part)
            unique.append(part)
    if not unique:
        return None

    raws = engine.store.all_raw(user_id)
    timestamp = max((int(x.get("timestamp") or 0) for x in raws), default=0)
    return {
        "id": f"p12-summary:{user_id}",
        "content": "历史记忆摘要：" + "\n".join(unique[:28]),
        "role": "context",
        "timestamp": timestamp,
        "user_id": user_id,
        "session_id": "__global__",
        "score": 1.01,
        "source": "p12_summary_compat",
        "memory_type": "summary",
        "status": "active",
        "valid_from": timestamp,
        "valid_to": None,
        "metadata": {
            "p12_channel": "summary_compat",
            "p12_compression": True,
            "auxiliary_only": True,
        },
    }


def build_evidence_dossier(
    query: str,
    raws: list[dict[str, Any]],
    *,
    max_messages: int = 24,
    max_per_session: int = 6,
) -> dict[str, Any] | None:
    """Build a lossless, provenance-rich retrieval item for global questions."""
    if not raws or not is_global_history_query(query):
        return None

    rows = [r for r in raws if str(r.get("content") or "").strip()]
    if not rows:
        return None

    query_tokens = _tokens(query)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("session_id") or "")].append(row)

    for group in groups.values():
        group.sort(key=lambda r: (int(r.get("timestamp") or 0), str(r.get("id") or "")))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def relevance(row: dict[str, Any]) -> float:
        overlap = len(query_tokens & _tokens(row.get("content", "")))
        ts = int(row.get("timestamp") or 0)
        return overlap * 10.0 + (ts / 10**13)

    for session_id, group in sorted(
        groups.items(),
        key=lambda item: max(int(r.get("timestamp") or 0) for r in item[1]),
        reverse=True,
    ):
        anchors = sorted(group, key=relevance, reverse=True)[:max_per_session]
        if anchors:
            candidates = anchors + [group[0], group[-1]]
            for row in candidates:
                rid = str(row.get("id") or "")
                if rid and rid not in selected_ids:
                    selected.append(row)
                    selected_ids.add(rid)
                    break
        if len(selected) >= max_messages:
            break

    remaining = [r for r in rows if str(r.get("id") or "") not in selected_ids]
    remaining.sort(key=relevance, reverse=True)
    for row in remaining:
        if len(selected) >= max_messages:
            break
        sid = str(row.get("session_id") or "")
        if sum(1 for x in selected if str(x.get("session_id") or "") == sid) >= max_per_session:
            continue
        selected.append(row)
        selected_ids.add(str(row.get("id") or ""))

    selected.sort(key=lambda r: (int(r.get("timestamp") or 0), str(r.get("id") or "")))
    if not selected:
        return None

    blocks = []
    for row in selected:
        ts = int(row.get("timestamp") or 0)
        blocks.append(
            f"[session={row.get('session_id') or '-'} timestamp={ts} role={row.get('role') or 'user'}] "
            f"{str(row.get('content') or '').strip()}"
        )

    latest_ts = max(int(r.get("timestamp") or 0) for r in selected)
    return {
        "id": "p12-history-dossier:" + str(abs(hash(tuple(selected_ids)))),
        "content": "GLOBAL MEMORY EVIDENCE DOSSIER\n" + "\n".join(blocks),
        "role": "context",
        "timestamp": latest_ts,
        "user_id": selected[0].get("user_id"),
        "session_id": "__global__",
        "score": 0.12,
        "source": "p12_history_dossier",
        "memory_type": "session",
        "status": "active",
        "valid_from": int(selected[0].get("timestamp") or 0),
        "valid_to": None,
        "metadata": {
            "source_message_ids": [str(r.get("id")) for r in selected if r.get("id")],
            "session_ids": sorted({str(r.get("session_id") or "") for r in selected}),
            "lossless": True,
            "global_history": True,
        },
    }


def _install() -> None:
    from .engine import MemoryEngine

    if getattr(MemoryEngine.search, "_p12_installed", False):
        return

    original_search = MemoryEngine.search

    def search_with_history_dossier(self, request):
        rows = list(original_search(self, request))
        query = (request.query or request.question or "").strip()
        if not is_global_history_query(query):
            return rows

        raws = self.store.all_raw(request.user_id)
        if request.start_time is not None:
            raws = [r for r in raws if int(r.get("timestamp") or 0) >= request.start_time]
        if request.end_time is not None:
            raws = [r for r in raws if int(r.get("timestamp") or 0) <= request.end_time]

        dossier = build_evidence_dossier(query, raws, max_messages=24, max_per_session=6)
        summary = _build_compat_summary(self, request.user_id)
        if dossier is None and summary is None:
            return rows

        limit = max(1, int(request.top_k))
        existing_ids = {str(r.get("id")) for r in rows}

        # Both additions are auxiliary. Raw/P10 evidence remains canonical.
        # Put the P9-compatible summary first so existing summary queries keep
        # their expected memory_type while the lossless dossier remains visible.
        additions = [x for x in (summary, dossier) if x is not None]
        for item in reversed(additions):
            if str(item.get("id")) not in existing_ids:
                rows.insert(0, item)
                existing_ids.add(str(item.get("id")))

        return rows[:limit]

    search_with_history_dossier._p12_installed = True
    MemoryEngine.search = search_with_history_dossier


_install()
