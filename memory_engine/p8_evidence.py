"""P8 evidence-completeness retrieval and query-time state policy.

P8 is deliberately additive. It keeps the raw/structured memories immutable,
uses the existing hybrid retriever as the source of truth, and adds bounded
retrieval channels inspired by ReFind-style iterative evidence search and
bounded graph traversal. The graph only selects stored evidence; it never
answers the question.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .query_analyzer import QueryPlan


class EvidenceCompletenessPool:
    """Recover evidence that a single semantic query can miss."""

    MAX_REQUIREMENTS = 4
    REQUIREMENT_TOP_K = 8
    MAX_TARGETED_TOP_K = 8
    MAX_BRIDGE_RELATIONS = 32
    MAX_BRIDGE_ENTITIES = 16
    BRIDGE_HOPS = 2

    # Deliberately tiny: P8 is a recall/completeness channel, not a second
    # reranker. The existing hybrid/reranker pipeline remains authoritative.
    REQUIREMENT_SCORE = 0.010
    TARGETED_SCORE = 0.011
    BRIDGE_SCORE = 0.012

    CURRENT_MARKERS = (
        "现在", "目前", "当前", "最新", "如今", "现居",
        "currently", "right now", "at present", "presently", "latest", "current", "now",
    )
    TEMPORAL_WORDS = (
        "以前", "之前", "后来", "之后", "最近", "历史", "过去", "去年", "前年", "今年",
        "上个月", "本月", "上周", "本周", "下周", "昨天", "今天", "明天",
        "before", "after", "previously", "later", "earlier", "recently", "yesterday",
        "tomorrow", "last week", "this week", "next week", "last month", "this month",
        "last year", "this year", "ago",
    )

    def __init__(self, engine: Any):
        self.engine = engine

    @staticmethod
    def _clone(row: dict, score: float, **metadata) -> dict:
        item = dict(row)
        item["score"] = float(score)
        md = dict(item.get("metadata", {}) or {})
        md.update(metadata)
        item["metadata"] = md
        return item

    @staticmethod
    def _entities_from_rows(rows: list[dict]) -> list[str]:
        values: list[str] = []
        ignored = {"user", "我", "用户", "none", "null"}
        for row in rows:
            md = row.get("metadata", {}) or {}
            for key in ("subject", "object", "value", "entity"):
                value = str(md.get(key) or "").strip()
                if value and value.lower() not in ignored and value not in values:
                    values.append(value)
        return values[:16]

    @classmethod
    def _is_current_query(cls, query: str) -> bool:
        lower = query.lower()
        return any(marker in lower for marker in cls.CURRENT_MARKERS)

    @classmethod
    def _is_temporal_query(cls, query: str) -> bool:
        lower = query.lower()
        return any(marker in lower for marker in cls.TEMPORAL_WORDS)

    def _requirements(self, plan: QueryPlan) -> list[dict]:
        requirements = self.engine.query_analyzer.evidence_requirements(plan)
        return sorted(
            requirements,
            key=lambda x: (-int(x.get("priority", 0)), str(x.get("id", ""))),
        )[: self.MAX_REQUIREMENTS]

    def _requirement_candidates(self, request: Any, plan: QueryPlan) -> list[dict]:
        output: list[dict] = []
        seen: set[str] = set()
        for req in self._requirements(plan):
            q = str(req.get("query") or "").strip()
            if not q:
                continue
            try:
                rows = self.engine.hybrid.candidates(
                    user_id=request.user_id,
                    query=q,
                    top_k=self.REQUIREMENT_TOP_K,
                    include_history=request.include_history or plan.temporal,
                    session_id=request.session_id,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    memory_types=request.memory_types,
                    memory_type_hint=plan.memory_type_hint,
                    temporal_relation=plan.temporal_relation,
                    relation_hint=plan.relation_hint,
                    sparse_query=q,
                    predicate_hint=plan.predicate_hint,
                    intent_hint=plan.intent_hint,
                )
            except Exception:
                continue
            for row, score in rows:
                cid = str(row.get("id") or "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                output.append(self._clone(
                    row,
                    self.REQUIREMENT_SCORE,
                    p8_channel="requirement",
                    p8_requirement_id=req.get("id"),
                    p8_requirement_kind=req.get("kind"),
                    p8_requirement_score=float(score),
                ))
        return output

    def _targeted_candidates(self, request: Any, plan: QueryPlan) -> list[dict]:
        """Run a small second retrieval for state/time questions.

        This is the ReFind-style "search again with a more specific query"
        part of P8, but deterministic and bounded. It never replaces the
        original query and never hard-filters the main candidate pool.
        """
        query = plan.original.strip()
        if not query:
            return []
        current = self._is_current_query(query)
        temporal = self._is_temporal_query(query) or plan.temporal
        if not (current or temporal):
            return []

        terms = [query]
        if plan.predicate_hint:
            terms.append(str(plan.predicate_hint))
        if current:
            terms.extend(["current", "latest", "active", "valid"])
        if temporal:
            terms.extend(["before", "after", "previous", "historical", "date", "time"])
        targeted = " ".join(dict.fromkeys(x for x in terms if x))

        try:
            rows = self.engine.hybrid.candidates(
                user_id=request.user_id,
                query=targeted,
                top_k=self.MAX_TARGETED_TOP_K,
                include_history=request.include_history or temporal,
                session_id=request.session_id,
                start_time=request.start_time,
                end_time=request.end_time,
                memory_types=request.memory_types,
                memory_type_hint=plan.memory_type_hint,
                temporal_relation=plan.temporal_relation,
                relation_hint=plan.relation_hint,
                sparse_query=targeted,
                predicate_hint=plan.predicate_hint,
                intent_hint=plan.intent_hint,
            )
        except Exception:
            return []

        channel = "state" if current else "temporal"
        output = []
        for row, score in rows:
            output.append(self._clone(
                row,
                self.TARGETED_SCORE,
                p8_channel=channel,
                p8_targeted_score=float(score),
                p8_predicate=plan.predicate_hint,
            ))
        return output

    def _bridge_candidates(self, request: Any, seed_rows: list[dict], plan: QueryPlan) -> list[dict]:
        """Recover a bounded 1-2 hop relation path around query entities."""
        if not plan.multi_hop:
            return []

        entities = self._entities_from_rows(seed_rows)
        for req in self._requirements(plan):
            value = str(req.get("entity") or "").strip()
            if value and value not in entities:
                entities.append(value)
        entities = entities[: self.MAX_BRIDGE_ENTITIES]
        if not entities:
            return []

        relations = self.engine.store.relations(request.user_id)
        adjacency: dict[str, list[dict]] = defaultdict(list)
        for rel in relations:
            subject = str(rel.get("subject") or "").strip()
            obj = str(rel.get("object") or "").strip()
            if subject:
                adjacency[subject].append(rel)
            if obj:
                adjacency[obj].append(rel)

        # Soft predicate routing: prefer relations that look like the answer
        # shape, but never discard another edge because the query classifier
        # was imperfect.
        query_lower = plan.original.lower()
        predicate_terms = []
        if any(x in query_lower for x in ("headquarter", "总部")):
            predicate_terms += ["headquarters", "located", "location", "总部", "位于"]
        if any(x in query_lower for x in ("work", "job", "occupation", "公司")):
            predicate_terms += ["works_at", "employed", "occupation", "company", "工作", "公司"]
        if any(x in query_lower for x in ("friend", "coworker", "同事", "朋友")):
            predicate_terms += ["friend", "coworker", "同事", "朋友"]
        if any(x in query_lower for x in ("why", "cause", "because", "导致", "原因")):
            predicate_terms += ["causes", "reason", "cause", "导致", "原因"]

        output: list[dict] = []
        seen: set[str] = set()
        frontier = list(entities)
        visited_entities = set(frontier)

        for hop in range(1, self.BRIDGE_HOPS + 1):
            next_frontier: list[str] = []
            for entity in frontier:
                candidates = list(adjacency.get(entity, ()))
                candidates.sort(key=lambda rel: (
                    0 if any(term in str(rel.get("predicate") or "").lower() or
                               term.lower() in str(rel.get("content") or "").lower()
                               for term in predicate_terms) else 1,
                    -int(rel.get("timestamp") or 0),
                ))
                for rel in candidates:
                    rid = str(rel.get("id") or "")
                    if not rid or rid in seen:
                        continue
                    seen.add(rid)
                    subject = str(rel.get("subject") or "").strip()
                    obj = str(rel.get("object") or "").strip()
                    matched = any(
                        term.lower() in str(rel.get("predicate") or "").lower()
                        or term.lower() in str(rel.get("content") or "").lower()
                        for term in predicate_terms
                    )
                    output.append({
                        "id": rid,
                        "content": rel.get("content", ""),
                        "role": "relation",
                        "timestamp": rel.get("timestamp", 0),
                        "user_id": request.user_id,
                        "session_id": "",
                        "score": self.BRIDGE_SCORE + (0.001 if matched else 0.0),
                        "source": "p8_bounded_bridge",
                        "memory_type": "relation",
                        "status": rel.get("status", "active"),
                        "valid_from": rel.get("timestamp", 0),
                        "valid_to": None,
                        "metadata": {
                            "subject": subject,
                            "predicate": rel.get("predicate"),
                            "object": obj,
                            "p8_channel": "bridge",
                            "p8_hop": hop,
                            "p8_path_start": entity,
                            "p8_path_edge": f"{subject} -> {rel.get('predicate')} -> {obj}",
                            "p8_predicate_match": matched,
                        },
                    })
                    for value in (subject, obj):
                        if value and value not in visited_entities and len(next_frontier) < self.MAX_BRIDGE_ENTITIES:
                            visited_entities.add(value)
                            next_frontier.append(value)
                    if len(output) >= self.MAX_BRIDGE_RELATIONS:
                        return output
            frontier = next_frontier
            if not frontier:
                break
        return output

    def collect(self, request: Any, plan: QueryPlan, seed_rows: list[dict]) -> list[dict]:
        extra = self._requirement_candidates(request, plan)
        extra.extend(self._targeted_candidates(request, plan))
        extra.extend(self._bridge_candidates(request, seed_rows, plan))

        dedup: dict[str, dict] = {}
        for item in extra:
            cid = str(item.get("id") or "")
            if cid and cid not in dedup:
                dedup[cid] = item
        return list(dedup.values())

    @staticmethod
    def promote_complementary_paths(rows: list[dict], top_k: int, multi_hop: bool) -> list[dict]:
        """Small deterministic post-rerank diversity guard.

        Cross-encoders tend to reward a single locally relevant edge. For a
        multi-hop question the answer often needs two complementary edges.
        We therefore preserve at least one bridge item from each observed hop
        when those items already survived the reranker. This does not invent
        or force missing evidence into the result set.
        """
        if not multi_hop or len(rows) <= 1:
            return rows
        by_hop: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            md = row.get("metadata", {}) or {}
            if md.get("p8_channel") == "bridge":
                try:
                    by_hop[int(md.get("p8_hop", 0))].append(row)
                except (TypeError, ValueError):
                    pass
        if len(by_hop) < 2:
            return rows

        selected = []
        selected_ids = set()
        for hop in sorted(by_hop):
            best = max(by_hop[hop], key=lambda x: float(x.get("score", 0.0)))
            selected.append(best)
            selected_ids.add(str(best.get("id") or ""))
        remaining = [r for r in rows if str(r.get("id") or "") not in selected_ids]
        remaining.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return (selected + remaining)[:top_k]


def install_p8() -> None:
    """Install P8 once while keeping the original engine as the final authority."""
    from . import engine as engine_module

    if getattr(engine_module.MemoryEngine, "_p8_installed", False):
        return

    original_search = engine_module.MemoryEngine.search

    def search_with_p8(self, request):
        original_candidates = self.hybrid.candidates
        injected = {"done": False}

        def candidates_with_p8(*args, **kwargs):
            rows = original_candidates(*args, **kwargs)
            if injected["done"]:
                return rows
            injected["done"] = True

            query = (request.query or request.question or "").strip()
            if not query:
                return rows
            try:
                latest = self.store.all_raw(request.user_id)
                reference_ts = latest[0]["timestamp"] if latest else None
                plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
                pool = EvidenceCompletenessPool(self)
                seed_rows = [dict(row) for row, _score in rows]
                extra = pool.collect(request, plan, seed_rows)
                existing = {str(row.get("id") or "") for row, _score in rows}
                rows = list(rows)
                rows.extend(
                    (item, float(item.get("score", 0.0)))
                    for item in extra
                    if str(item.get("id") or "") not in existing
                )
            except Exception:
                return rows
            return rows

        self.hybrid.candidates = candidates_with_p8
        try:
            result = original_search(self, request)
        finally:
            self.hybrid.candidates = original_candidates

        query = (request.query or request.question or "").strip()
        try:
            latest = self.store.all_raw(request.user_id)
            reference_ts = latest[0]["timestamp"] if latest else None
            plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
            result = EvidenceCompletenessPool.promote_complementary_paths(
                list(result), request.top_k, plan.multi_hop
            )

            # Read-time current-state policy: never delete historical facts;
            # only move the newest active fact matching the requested predicate
            # ahead of older/conflicting candidates. This mirrors the
            # append-only/read-time approach used by strong memory baselines.
            if not request.include_history and EvidenceCompletenessPool._is_current_query(query):
                predicate = plan.predicate_hint
                if predicate:
                    result.sort(key=lambda item: (
                        1 if item.get("memory_type") == "fact" and
                        item.get("status") == "active" and
                        (item.get("metadata", {}) or {}).get("predicate") == predicate else 0,
                        1 if item.get("status") == "active" else 0,
                        int(item.get("valid_from", item.get("timestamp", 0)) or 0),
                        float(item.get("score", 0.0)),
                    ), reverse=True)
        except Exception:
            pass
        return result[:request.top_k]

    engine_module.MemoryEngine.search = search_with_p8
    engine_module.MemoryEngine._p8_installed = True


install_p8()
