from __future__ import annotations

import re

from . import analyzer as analyzer_module
from . import governance as governance_module
from . import query_analyzer as query_module
from .models import new_id


CURRENT_MARKERS_EN = (
    "currently", "right now", "at present", "presently", "as of now",
    "these days", "today", "latest", "current", "now",
)

TEMPORAL_MARKERS_EN = (
    "now", "currently", "right now", "at present", "presently", "today",
    "yesterday", "tomorrow", "last week", "this week", "next week",
    "last month", "this month", "last year", "this year", "previously",
    "before", "after", "later", "earlier", "recently", "ago",
)


class P7MemoryAnalyzer(analyzer_module.MemoryAnalyzer):
    """Extend deterministic extraction without replacing the P6 parser."""

    CAUSAL_PATTERNS = analyzer_module.MemoryAnalyzer.CAUSAL_PATTERNS + (
        re.compile(r"because\s+(.{1,80}?)\s*(?:,|;)?\s*(?:so|therefore|thus|hence)\s+(.{1,80})", re.I),
        re.compile(r"(.{1,60}?)\s+(?:caused|causes|led to|resulted in|triggered)\s+(.{1,60})", re.I),
        re.compile(r"(.{1,60}?)\s+(?:was|is)\s+because\s+(.{1,60})", re.I),
    )

    EVENT_WORDS = analyzer_module.MemoryAnalyzer.EVENT_WORDS + (
        "moved", "graduated", "started", "ended", "joined", "left", "married",
        "divorced", "traveled", "visited", "attended", "bought", "completed",
        "started working", "started a job", "quit", "returned",
    )


class P7QueryAnalyzer(query_module.QueryAnalyzer):
    """Make English AML temporal/current-state questions first-class."""

    TEMPORAL_MARKERS = query_module.QueryAnalyzer.TEMPORAL_MARKERS + TEMPORAL_MARKERS_EN

    @staticmethod
    def infer_predicate(q: str) -> str | None:
        base = query_module.QueryAnalyzer.infer_predicate(q)
        if base:
            return base
        lower = q.lower()
        if any(x in lower for x in ("where do i live", "where does", "where did", "live", "reside", "residence", "home")):
            return "residence"
        if any(x in lower for x in ("job", "occupation", "work", "career", "profession")):
            return "occupation"
        if any(x in lower for x in ("birthday", "date of birth", "born")):
            return "birthday"
        if any(x in lower for x in ("name", "called")):
            return "name"
        if any(x in lower for x in ("dislike", "hate", "don't like", "do not like")):
            return "dislike"
        if any(x in lower for x in ("like", "favorite", "favourite", "prefer", "preference")):
            return "like"
        return None

    @staticmethod
    def infer_memory_type(q: str) -> str | None:
        base = query_module.QueryAnalyzer.infer_memory_type(q)
        if base:
            return base
        lower = q.lower()
        if any(x in lower for x in ("when did", "when was", "what happened", "what year", "what date")):
            return "event"
        if any(x in lower for x in ("why", "because", "caused", "cause", "led to", "resulted in", "relationship", "friend", "coworker", "company")):
            return "relation"
        if any(x in lower for x in ("like", "favorite", "prefer", "dislike")):
            return "fact"
        return None

    @staticmethod
    def infer_intent(q: str, memory_type_hint: str | None) -> str | None:
        base = query_module.QueryAnalyzer.infer_intent(q, memory_type_hint)
        if base:
            return base
        if any(x in q.lower() for x in ("why", "because", "caused", "led to", "resulted in")):
            return "causal"
        return memory_type_hint


class P7MemoryGovernance(governance_module.MemoryGovernance):
    """Keep newer knowledge current while preserving out-of-order evidence."""

    def accept_fact(self, fact):
        current = self.store.find_current_fact(fact.user_id, fact.subject, fact.predicate)
        if current and current["object"] != fact.object:
            current_ts = int(current["timestamp"] or current["valid_from"] or 0)
            fact_ts = int(getattr(fact, "timestamp", 0) or 0)
            if fact_ts < current_ts:
                fact.conflict_status = "historical_conflict"
                fact.conflict_group_id = f"late:{fact.user_id}:{fact.predicate}:{fact_ts}"
                self.store.insert_fact(fact)
                self.store.insert_conflict_log({
                    "id": new_id("conflict_log"),
                    "user_id": fact.user_id,
                    "predicate": fact.predicate,
                    "old_fact_id": current["id"],
                    "new_fact_id": fact.id,
                    "old_object": current["object"],
                    "new_object": fact.object,
                    "resolution": "preserved_historical_fact",
                    "reason": f"out_of_order_timestamp={fact_ts} < current={current_ts}",
                    "created_at": fact_ts,
                })
                return True, "preserved_historical_fact"
        return super().accept_fact(fact)


def install_p7():
    """Install P7 classes before MemoryEngine imports its collaborators."""
    analyzer_module.MemoryAnalyzer = P7MemoryAnalyzer
    query_module.QueryAnalyzer = P7QueryAnalyzer
    governance_module.MemoryGovernance = P7MemoryGovernance

    # The engine's current-state prioritizer uses explicit Chinese markers.
    # Normalize English current-state questions into the same retrieval signal.
    try:
        from . import engine as engine_module
    except ImportError:
        return
    if getattr(engine_module.MemoryEngine, "_p7_installed", False):
        return

    original_search = engine_module.MemoryEngine.search

    def search_with_p7(self, request):
        query = (request.query or request.question or "").strip()
        lower = query.lower()
        is_current = any(marker in lower for marker in CURRENT_MARKERS_EN)
        original_query = request.query
        if is_current and "现在" not in query:
            request.query = query + " 当前 现在"
        try:
            return original_search(self, request)
        finally:
            request.query = original_query

    engine_module.MemoryEngine.search = search_with_p7
    engine_module.MemoryEngine._p7_installed = True


install_p7()
