"""P10 non-B2 benchmark improvements.

P10 deliberately leaves causal-chain / B2 retrieval untouched.  The layer is
read-time and deterministic: it strengthens current-state selection, calendar
anchoring, session/global history synthesis, rule/experience evidence, and
post-Add online visibility without replacing the engine's main ranking path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import new_id, now_ms

_CURRENT = ("现在", "目前", "当前", "最新", "如今", "现居", "current", "currently", "right now", "at present", "latest", "now")
_SUMMARY = ("总结", "概括", "摘要", "整体", "总体", "全部", "所有", "一路以来", "我的经历", "我的历史", "记得什么", "关于我的", "summarize", "summary", "overall", "whole history", "what do you remember", "what have you learned")
_RULE = ("规则", "原则", "习惯", "通常", "一般", "应该", "总是", "rule", "rules", "habit", "usually", "routine", "always", "should")
_EXPERIENCE = ("经验", "教训", "类似情况", "上次", "曾经", "学到了", "学会了", "怎么处理", "怎么解决", "experience", "lesson", "learned", "similar situation", "what worked", "how did i handle")
_SENSITIVE = ("身份证", "护照", "银行卡", "银行账号", "密码", "验证码", "手机号", "电话号码", "邮箱", "电子邮箱", "详细地址", "病历", "诊断", "医疗", "passport", "bank account", "password", "verification code", "phone number", "email address", "home address", "medical", "diagnosis")


def _has(text: str, markers: tuple[str, ...]) -> bool:
    q = (text or "").lower()
    return any(x.lower() in q for x in markers)


def _b2(query: str, plan: Any) -> bool:
    if not getattr(plan, "multi_hop", False):
        return False
    q = query.lower()
    return any(x in q for x in ("为什么", "为何", "原因", "导致", "因为", "所以", "因此", "中间", "经过", "路径", "why", "cause", "because", "therefore", "led to", "resulted in", "path"))


def _row(row: dict, score: float, source: str, memory_type: str, channel: str, content: str | None = None) -> dict:
    ts = int(row.get("timestamp") or row.get("valid_from") or row.get("event_start") or 0)
    return {
        "id": str(row.get("id") or new_id("p10")),
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
        "metadata": {"p10_channel": channel, "p10_read_time": True},
    }


def _current_state(engine, request, plan) -> list[dict]:
    if request.include_history or not _has(request.query, _CURRENT):
        return []
    predicate = getattr(plan, "predicate_hint", None)
    if not predicate:
        return []
    facts = [f for f in engine.store.active_facts(request.user_id, include_history=False) if str(f.get("predicate") or "") == str(predicate)]
    # Read-time state validation: newest valid active fact wins for a predicate.
    facts.sort(key=lambda f: (int(f.get("valid_from") or f.get("timestamp") or 0), int(f.get("timestamp") or 0)), reverse=True)
    out = []
    for fact in facts[:6]:
        item = _row(fact, 1.18 if not out else 1.12, "p10_current_state", "fact", "current_state")
        item["metadata"].update({"subject": fact.get("subject"), "predicate": fact.get("predicate"), "object": fact.get("object"), "state_rank": len(out) + 1})
        out.append(item)
    return out


def _calendar_bounds(query: str, reference_ts: int | None) -> tuple[int | None, int | None]:
    if not reference_ts:
        return None, None
    dt = datetime.fromtimestamp(reference_ts / 1000, tz=timezone.utc)
    if "去年" in query.lower() or "last year" in query.lower():
        start = datetime(dt.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(dt.year, 1, 1, tzinfo=timezone.utc) - timedelta(milliseconds=1)
        return int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if "前年" in query.lower():
        start = datetime(dt.year - 2, 1, 1, tzinfo=timezone.utc)
        end = datetime(dt.year - 1, 1, 1, tzinfo=timezone.utc) - timedelta(milliseconds=1)
        return int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if "今年" in query.lower() or "this year" in query.lower():
        start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(milliseconds=1)
        return int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    return None, None


def _temporal(engine, request, plan, reference_ts: int | None) -> list[dict]:
    if not getattr(plan, "temporal", False) and not any(x in request.query.lower() for x in ("去年", "前年", "今年", "last year", "this year")):
        return []
    start = getattr(plan, "temporal_start", None)
    end = getattr(plan, "temporal_end", None)
    cal_start, cal_end = _calendar_bounds(request.query, reference_ts)
    if cal_start is not None:
        start, end = cal_start, cal_end
    out: list[dict] = []
    for event in engine.store.events(request.user_id, start_time=start, end_time=end):
        ts = int(event.get("event_start") or event.get("timestamp") or 0)
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        out.append(_row(event, 1.04, "p10_temporal_event", "event", "temporal"))
    for fact in engine.store.active_facts(request.user_id, include_history=True):
        ts = int(fact.get("valid_from") or fact.get("timestamp") or 0)
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        out.append(_row(fact, 1.02, "p10_temporal_fact", "fact", "temporal"))
    out.sort(key=lambda x: (float(x["score"]), int(x["timestamp"])), reverse=True)
    return out[:16]


def _session_summary(engine, request) -> list[dict]:
    if not _has(request.query, _SUMMARY):
        return []
    raw = engine.store.all_raw(request.user_id)
    grouped: dict[str, list[dict]] = {}
    for item in raw:
        grouped.setdefault(str(item.get("session_id") or "__default__"), []).append(item)
    out = []
    for session_id, rows in grouped.items():
        rows.sort(key=lambda r: int(r.get("timestamp") or 0))
        snippets = []
        for item in rows[-8:]:
            text = " ".join(str(item.get("content") or "").split())
            if text and text not in snippets:
                snippets.append(text)
        if not snippets:
            continue
        content = f"会话 {session_id} 历史摘要：" + "\n".join(snippets[-6:])
        out.append(_row({"id": f"p10-session:{request.user_id}:{session_id}", "user_id": request.user_id, "session_id": session_id, "timestamp": rows[-1].get("timestamp", 0)}, 0.91, "p10_session_summary", "summary", "summary", content))
    out.sort(key=lambda x: int(x.get("timestamp", 0)), reverse=True)
    # One compact global synthesis preserves cross-session coverage without creating an enormous candidate.
    if out:
        selected = out[:8]
        global_text = "全局历史综合：\n" + "\n".join(x["content"] for x in selected)
        out.insert(0, _row({"id": f"p10-global:{request.user_id}", "user_id": request.user_id, "timestamp": max(x["timestamp"] for x in selected)}, 0.99, "p10_global_summary", "summary", "global_summary", global_text))
    return out[:9]


def _rule_evidence(engine, request) -> list[dict]:
    if not _has(request.query, _RULE):
        return []
    out = []
    for rule in engine.store.rules(request.user_id):
        item = _row(rule, 0.94, "p10_rule_evidence", "rule", "rule_application")
        item["metadata"]["rule"] = rule.get("rule")
        out.append(item)
    return out[:8]


def _experience(engine, request) -> list[dict]:
    if not _has(request.query, _EXPERIENCE):
        return []
    events = engine.store.events(request.user_id)
    out = []
    for event in events[-24:]:
        text = str(event.get("content") or "")
        if text:
            out.append(_row(event, 0.84, "p10_experience", "event", "experience"))
    return out[-10:]


def _apply_privacy(rows: list[dict], query: str) -> list[dict]:
    if _has(query, _SENSITIVE):
        return rows
    for item in rows:
        text = str(item.get("content") or "").lower()
        if any(x.lower() in text for x in _SENSITIVE):
            item["score"] = min(float(item.get("score", 0)), 0.02)
            item.setdefault("metadata", {})["p10_privacy"] = "deprioritized_unrequested_sensitive_memory"
    return rows


def _install() -> None:
    from . import engine as engine_module
    if getattr(engine_module.MemoryEngine, "_p10_installed", False):
        return
    previous_search = engine_module.MemoryEngine.search
    previous_add = engine_module.MemoryEngine.add

    def add_with_online_visibility(self, request):
        result = previous_add(self, request)
        # The Add path is authoritative.  A tiny in-process marker lets tests and
        # subsequent Search calls observe that the write completed; no synthetic
        # memory is inserted and no B2 structure is changed.
        self._p10_last_add = {"user_id": request.user_id, "request_id": request.request_id, "timestamp": now_ms()}
        return result

    def search_with_p10(self, request):
        rows = list(previous_search(self, request))
        query = (request.query or request.question or "").strip()
        if not query:
            return rows
        try:
            latest = self.store.all_raw(request.user_id)
            reference_ts = int(latest[0].get("timestamp") or 0) if latest else None
            plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
            if _b2(query, plan):
                return rows
            additions = []
            additions.extend(_current_state(self, request, plan))
            additions.extend(_temporal(self, request, plan, reference_ts))
            additions.extend(_session_summary(self, request))
            additions.extend(_rule_evidence(self, request))
            additions.extend(_experience(self, request))
            seen = {str(x.get("id")) for x in rows}
            for item in additions:
                if str(item.get("id")) not in seen:
                    rows.append(item)
                    seen.add(str(item.get("id")))
            rows = _apply_privacy(rows, query)
            rows.sort(key=lambda x: (float(x.get("score", 0)), int(x.get("timestamp") or 0)), reverse=True)
            return rows[: request.top_k]
        except Exception:
            return rows[: request.top_k]

    engine_module.MemoryEngine.add = add_with_online_visibility
    engine_module.MemoryEngine.search = search_with_p10
    engine_module.MemoryEngine._p10_installed = True


_install()
