"""P14 G2: grounded rule-application retrieval.

This layer is intentionally isolated from P10/P13. It does not change rule
storage or the general ranking path. For explicit rule/application questions,
it turns stored if/then rules into compact, grounded evidence and optionally
attaches matching raw/fact context. No new rule or conclusion is invented.
"""

from __future__ import annotations

import re
from typing import Any

from .models import new_id


_RULE_INTENT_MARKERS = (
    "规则", "原则", "习惯", "通常", "一般", "应该", "应当", "总是",
    "如果", "那么", "按照", "根据", "遵循", "怎么办", "怎么处理", "怎么做",
    "rule", "rules", "habit", "usually", "routine", "always", "should",
    "if", "then", "according to", "how should", "what should i do",
)

_APPLICATION_MARKERS = (
    "如果", "当", "遇到", "情况下", "怎么办", "怎么处理", "怎么做", "应该",
    "应当", "按照", "根据", "遵循", "如何", "if", "when", "should",
    "according to", "how should", "what should",
)

_STOP = {
    "我", "你", "他", "她", "它", "我们", "他们", "这个", "那个", "什么", "怎么",
    "如何", "为什么", "是否", "可以", "能否", "请问", "一下", "一下子", "我的", "你的",
    "the", "a", "an", "i", "you", "we", "they", "it", "what", "how", "why", "can",
    "should", "would", "could", "do", "does", "did", "according", "to", "my", "your",
}


def _has_marker(query: str, markers: tuple[str, ...]) -> bool:
    q = (query or "").lower()
    return any(m.lower() in q for m in markers)


def _terms(text: str) -> set[str]:
    q = (text or "").lower()
    words = re.findall(r"[a-z][a-z0-9_-]*|\d+(?:\.\d+)?", q)
    out = {w for w in words if len(w) > 1 and w not in _STOP}
    # Chinese evidence benefits from character bigrams because the project
    # deliberately avoids a heavyweight tokenizer in the deterministic path.
    zh = re.findall(r"[\u4e00-\u9fff]", q)
    for i in range(len(zh) - 1):
        bg = "".join(zh[i : i + 2])
        if bg not in {"如果", "那么", "应该", "怎么", "如何", "按照", "根据", "通常", "一般"}:
            out.add(bg)
    return out


def _parse_if_then(text: str) -> tuple[str, str] | None:
    s = " ".join(str(text or "").split()).strip(" 。！？!?；;,")
    patterns = (
        r"如果\s*(.+?)\s*(?:，|,)?\s*(?:就|那么|则|应该|应当)\s*(.+)",
        r"当\s*(.+?)\s*(?:，|,)?\s*(?:就|那么|则|应该|应当)\s*(.+)",
        r"if\s+(.+?)\s*,?\s*then\s+(.+)",
        r"if\s+(.+?)\s*,?\s*(?:you|i|we)\s+should\s+(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.I)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None


def _rule_score(query: str, rule_text: str) -> tuple[float, set[str]]:
    q_terms = _terms(query)
    r_terms = _terms(rule_text)
    if not q_terms or not r_terms:
        return 0.0, set()
    overlap = q_terms & r_terms
    # Favor coverage of the query: a rule should not be selected merely because
    # it shares one generic word such as "usually" or "should".
    coverage = len(overlap) / max(1, len(q_terms))
    precision = len(overlap) / max(1, len(r_terms))
    return 0.72 * coverage + 0.28 * precision, overlap


def _row(rule: dict[str, Any], score: float, content: str, channel: str, metadata: dict[str, Any] | None = None) -> dict:
    ts = int(rule.get("timestamp") or 0)
    item = {
        "id": f"p14-g2:{rule.get('id') or new_id('rule')}",
        "content": content,
        "role": "memory",
        "timestamp": ts,
        "user_id": rule.get("user_id"),
        "session_id": rule.get("session_id", ""),
        "score": float(score),
        "source": "p14_rule_application",
        "memory_type": "rule",
        "status": "active",
        "valid_from": ts,
        "valid_to": None,
        "metadata": {"p14_channel": channel, "grounded": True},
    }
    if metadata:
        item["metadata"].update(metadata)
    return item


def _application_evidence(engine, request) -> list[dict]:
    query = (request.query or request.question or "").strip()
    if not query or not _has_marker(query, _RULE_INTENT_MARKERS):
        return []

    rules = engine.store.rules(request.user_id)
    if not rules:
        return []

    candidates: list[tuple[float, dict, set[str]]] = []
    for rule in rules:
        rule_text = " ".join(str(rule.get(k) or "") for k in ("rule", "content")).strip()
        if not rule_text:
            continue
        score, overlap = _rule_score(query, rule_text)
        parsed = _parse_if_then(rule_text)
        # For explicit application questions, require either a meaningful
        # lexical match or a parsed condition that appears in the query.
        condition_match = False
        if parsed:
            condition_terms = _terms(parsed[0])
            q_terms = _terms(query)
            condition_match = len(condition_terms & q_terms) >= max(1, min(3, len(condition_terms)))
        if score >= 0.12 or condition_match:
            candidates.append((score + (0.18 if condition_match else 0.0), rule, overlap))

    candidates.sort(key=lambda x: (x[0], int(x[1].get("timestamp") or 0)), reverse=True)
    out: list[dict] = []
    for score, rule, overlap in candidates[:4]:
        rule_text = str(rule.get("rule") or rule.get("content") or "").strip()
        parsed = _parse_if_then(rule_text)
        if parsed:
            condition, action = parsed
            content = (
                f"已存规则：如果{condition}，则{action}。"
                f"\n当前问题与该规则条件存在匹配证据：{', '.join(sorted(overlap)[:8]) or '条件语义匹配'}。"
            )
            metadata = {
                "rule": rule.get("rule"),
                "condition": condition,
                "action": action,
                "matched_terms": sorted(overlap),
                "application": True,
            }
        else:
            content = f"已存行为规则：{rule_text}。当前问题与该规则存在相关证据。"
            metadata = {
                "rule": rule.get("rule"),
                "matched_terms": sorted(overlap),
                "application": True,
            }
        out.append(_row(rule, min(1.16, 0.98 + score), content, "rule_application", metadata))

    return out


def _install() -> None:
    from . import engine as engine_module
    if getattr(engine_module.MemoryEngine, "_p14_g2_installed", False):
        return

    previous_search = engine_module.MemoryEngine.search

    def search_with_g2(self, request):
        rows = list(previous_search(self, request))
        query = (request.query or request.question or "").strip()
        if not query:
            return rows
        try:
            # Do not touch the proven multi-hop path. G2 is evaluated only on
            # explicit rule/application intent.
            if getattr(request, "multi_hop", False):
                return rows
            additions = _application_evidence(self, request)
            if not additions:
                return rows[: request.top_k]
            seen = {str(x.get("id")) for x in rows}
            for item in additions:
                if str(item.get("id")) not in seen:
                    rows.append(item)
                    seen.add(str(item.get("id")))
            rows.sort(key=lambda x: (float(x.get("score", 0)), int(x.get("timestamp") or 0)), reverse=True)
            return rows[: request.top_k]
        except Exception:
            return rows[: request.top_k]

    engine_module.MemoryEngine.search = search_with_g2
    engine_module.MemoryEngine._p14_g2_installed = True


_install()
