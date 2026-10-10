"""P8 evidence-completeness retrieval layer.

P8 is intentionally retrieval-only. It does not generate answers, mutate
memory, or replace the existing reranker. It adds independent evidence
requirements and a bounded two-hop relation bridge to the candidate pool.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .query_analyzer import QueryPlan


class EvidenceCompletenessPool:
    """Collect additional evidence without changing answer generation."""

    MAX_REQUIREMENTS = 4
    REQUIREMENT_TOP_K = 8
    MAX_BRIDGE_RELATIONS = 24
    MAX_BRIDGE_ENTITIES = 16
    BRIDGE_HOPS = 2
    REQUIREMENT_SCORE = 0.010
    BRIDGE_SCORE = 0.012

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

    def _requirement_candidates(self, request: Any, plan: QueryPlan) -> list[dict]:
        requirements = self.engine.query_analyzer.evidence_requirements(plan)
        requirements = sorted(
            requirements,
            key=lambda x: (-int(x.get("priority", 0)), str(x.get("id", ""))),
        )[: self.MAX_REQUIREMENTS]

        output: list[dict] = []
        seen: set[str] = set()
        for req in requirements:
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
                output.append(
                    self._clone(
                        row,
                        self.REQUIREMENT_SCORE,
                        p8_channel="requirement",
                        p8_requirement_id=req.get("id"),
                        p8_requirement_kind=req.get("kind"),
                        p8_requirement_score=float(score),
                    )
                )
        return output

    def _bridge_candidates(self, request: Any, seed_rows: list[dict], plan: QueryPlan) -> list[dict]:
        """Recover bounded relation paths around planned/retrieved entities.

        The graph is only a selector. Returned items are original relation
        records from SQLite and therefore remain auditable evidence.
        """
        if not plan.multi_hop:
            return []

        entities = self._entities_from_rows(seed_rows)
        # Planned entities make the bridge useful even when the first dense
        # hit is raw prose and therefore has no structured subject/object.
        for req in self.engine.query_analyzer.evidence_requirements(plan):
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

        output: list[dict] = []
        seen: set[str] = set()
        frontier = list(entities)
        visited_entities = set(frontier)

        for hop in range(1, self.BRIDGE_HOPS + 1):
            next_frontier: list[str] = []
            for entity in frontier:
                for rel in adjacency.get(entity, ()):
                    rid = str(rel.get("id") or "")
                    if not rid or rid in seen:
                        continue
                    seen.add(rid)
                    subject = str(rel.get("subject") or "").strip()
                    obj = str(rel.get("object") or "").strip()
                    output.append({
                        "id": rid,
                        "content": rel.get("content", ""),
                        "role": "relation",
                        "timestamp": rel.get("timestamp", 0),
                        "user_id": request.user_id,
                        "session_id": "",
                        "score": self.BRIDGE_SCORE,
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
        extra.extend(self._bridge_candidates(request, seed_rows, plan))
        dedup: dict[str, dict] = {}
        for item in extra:
            cid = str(item.get("id") or "")
            if cid and cid not in dedup:
                dedup[cid] = item
        return list(dedup.values())


def install_p8() -> None:
    """Install the retrieval layer once, preserving the existing search."""
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
            return original_search(self, request)
        finally:
            self.hybrid.candidates = original_candidates

    engine_module.MemoryEngine.search = search_with_p8
    engine_module.MemoryEngine._p8_installed = True


install_p8()
