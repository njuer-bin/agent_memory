from __future__ import annotations

import re

from .engine import MemoryEngine
from .evidence_path import EvidencePathReconstructor
from .reasoning import MemoryReasoner


class ReasoningMemoryEngine(MemoryEngine):
    """P7 production wrapper around the stable P6 engine.

    P6 remains the source of truth for retrieval. This wrapper adds deterministic
    reasoning and, for multi-hop queries, a bounded evidence-path reconstruction
    pass that can recover raw provenance fragments missed by the initial Top-K.
    """

    def __init__(self, db_path="data/memory.db"):
        super().__init__(db_path)
        self.path_reconstructor = EvidencePathReconstructor(
            max_hops=3,
            beam_width=8,
            neighbors_per_node=6,
        )

    def search(self, request):
        rows = super().search(request)
        query = (request.query or request.question or "").strip()

        # Rebuild the lightweight plan only for the reasoning pass. P6's
        # internal plan remains untouched, keeping this change isolated.
        latest = self.store.all_raw(request.user_id)
        reference_ts = latest[0]["timestamp"] if latest else None
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)

        # AML frequently uses English month/year expressions that predate the
        # Chinese temporal marker list in QueryAnalyzer. Mark those queries as
        # temporal for the query-time evidence bonus without changing P6.
        if re.search(
            r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\s+(?:ago|later|after)\b",
            query,
            flags=re.I,
        ):
            plan.temporal = True

        # P9: reconstruct short evidence paths from the already retrieved
        # candidates plus persisted provenance links. Unlike a global graph
        # traversal, seeds are restricted to retrieval hits and expansion is
        # bounded by hop/beam/neighbor caps. Missing path nodes are recovered
        # from canonical raw messages before the final reasoning pass.
        if plan.multi_hop and len(rows) > 1:
            requirement_plans = self.query_analyzer.evidence_requirements(plan)
            link_rows = self.store.memory_links(request.user_id)
            relation_rows = self.store.relations(request.user_id)
            paths, promotions = self.path_reconstructor.reconstruct(
                query=query,
                candidates=rows,
                requirement_plans=requirement_plans,
                link_rows=link_rows,
                relation_rows=relation_rows,
            )

            existing_ids = {str(row.get("id")) for row in rows}
            missing_ids = [node_id for node_id in promotions if node_id not in existing_ids]
            if missing_ids:
                raw_rows = self.store.raw_by_ids(request.user_id, missing_ids[:16])
                for raw in raw_rows:
                    promotion = promotions.get(raw["id"], {})
                    rows.append({
                        "id": raw["id"],
                        "content": raw["content"],
                        "role": raw["role"],
                        "timestamp": raw["timestamp"],
                        "user_id": raw["user_id"],
                        "session_id": raw["session_id"],
                        "score": float(promotion.get("path_score", 0.18)) + 0.02,
                        "source": "evidence_path_provenance",
                        "memory_type": "raw",
                        "status": "active",
                        "valid_from": raw["timestamp"],
                        "valid_to": None,
                        "metadata": {
                            "source_message_ids": [raw["id"]],
                            "evidence_path": True,
                            "path_coverage": promotion.get("path_coverage", 0.0),
                            "path_depth": promotion.get("path_depth", 0),
                            "path_edge_types": promotion.get("path_edge_types", []),
                            "_evidence_requirements": [],
                        },
                    })

            # Promote path members, but do not let path reconstruction replace
            # the direct retrieval order wholesale. Coverage gets a bounded
            # bonus; semantic retrieval remains the primary signal.
            if paths:
                rows.sort(
                    key=lambda item: (
                        float((item.get("metadata") or {}).get("path_coverage", 0.0) or 0.0),
                        float(item.get("score", 0.0) or 0.0),
                    ),
                    reverse=True,
                )

        rows = MemoryReasoner.augment(
            query=query,
            plan=plan,
            ranked=rows,
            store=self.store,
            user_id=request.user_id,
        )

        # Re-apply the evidence-chain annotation after causal edges are added.
        # This is intentionally bounded to the returned P6 candidates plus a
        # small reasoning/path expansion; it does not re-run retrieval.
        if plan.multi_hop and len(rows) > 1:
            rows = self.evidence_chain.annotate(plan.rewritten, rows)

        return rows[: request.top_k]
