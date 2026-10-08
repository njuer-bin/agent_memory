from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidencePath:
    """A bounded query-time path over retrieved and provenance-connected evidence."""

    node_ids: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    covered_requirements: set[str] = field(default_factory=set)
    score: float = 0.0

    @property
    def depth(self) -> int:
        return max(0, len(self.node_ids) - 1)


class EvidencePathReconstructor:
    """Reconstruct small evidence paths with beam search and hard bounds."""

    def __init__(self, max_hops: int = 3, beam_width: int = 8, neighbors_per_node: int = 6):
        self.max_hops = max(1, max_hops)
        self.beam_width = max(1, beam_width)
        self.neighbors_per_node = max(1, neighbors_per_node)

    def _build_graph(self, link_rows: list[dict], relation_rows: list[dict]):
        graph = defaultdict(list)
        for link in link_rows:
            source = str(link.get("source_id") or "")
            target = str(link.get("target_id") or "")
            if not source or not target or source == target:
                continue
            confidence = float(link.get("confidence", 0.0) or 0.0)
            relation = str(link.get("relation") or "link")
            graph[source].append((target, relation, confidence))
            graph[target].append((source, relation, confidence))

        # Only connect structured relation endpoints when the persisted schema
        # exposes their IDs; never invent graph nodes from text values.
        for rel in relation_rows:
            subject_id = str(rel.get("subject_id") or "")
            object_id = str(rel.get("object_id") or "")
            if not subject_id or not object_id:
                continue
            confidence = 0.90 if rel.get("predicate") in {
                "causes", "headquarters", "belongs_to", "located_in"
            } else 0.72
            predicate = str(rel.get("predicate") or "relation")
            graph[subject_id].append((object_id, predicate, confidence))
            graph[object_id].append((subject_id, predicate, confidence))

        for node, edges in list(graph.items()):
            graph[node] = sorted(edges, key=lambda x: x[2], reverse=True)[: self.neighbors_per_node]
        return graph

    def reconstruct(
        self,
        query: str,
        candidates: list[dict],
        requirement_plans: list[dict],
        link_rows: list[dict],
        relation_rows: list[dict],
    ) -> tuple[list[EvidencePath], dict[str, dict]]:
        if not candidates:
            return [], {}

        graph = self._build_graph(link_rows, relation_rows)
        by_id = {str(item.get("id")): item for item in candidates if item.get("id")}
        req_ids = {str(r.get("id")) for r in requirement_plans if r.get("id")}
        if not graph:
            return [], {}

        # Seeds are direct retrieval results. This prevents unrelated graph
        # components from entering the search merely because they are connected.
        seeds = sorted(
            by_id.values(),
            key=lambda x: (
                len(set(x.get("_evidence_requirements", []) or [])),
                float(x.get("_requirement_score", 0.0) or 0.0),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )[:16]

        paths: list[EvidencePath] = []
        for seed in seeds:
            sid = str(seed.get("id"))
            covered = set(seed.get("_evidence_requirements", []) or []) & req_ids
            beam = [EvidencePath([sid], [], covered, float(seed.get("score", 0.0) or 0.0))]
            for _ in range(self.max_hops):
                expanded: list[EvidencePath] = []
                for path in beam:
                    node = path.node_ids[-1]
                    for other, edge_type, confidence in graph.get(node, [])[: self.neighbors_per_node]:
                        if other in path.node_ids:
                            continue
                        item = by_id.get(other)
                        new_cov = set(path.covered_requirements)
                        if item:
                            new_cov.update(item.get("_evidence_requirements", []) or [])
                            item_score = float(item.get("score", 0.0) or 0.0)
                        else:
                            # The node can be recovered from raw provenance by
                            # the caller. Give it a conservative score now.
                            item_score = 0.18
                        coverage = len(new_cov) / max(1, len(req_ids))
                        path_score = (
                            0.45 * item_score
                            + 0.35 * coverage
                            + 0.20 * confidence
                            - 0.03 * len(path.node_ids)
                        )
                        expanded.append(EvidencePath(
                            path.node_ids + [other],
                            path.edge_types + [edge_type],
                            new_cov,
                            path_score,
                        ))
                if not expanded:
                    break
                expanded.sort(key=lambda p: (len(p.covered_requirements), p.score), reverse=True)
                beam = expanded[: self.beam_width]
                paths.extend(beam)
                if req_ids and any(p.covered_requirements >= req_ids for p in beam):
                    break

        unique: dict[tuple[str, ...], EvidencePath] = {}
        for path in paths:
            key = tuple(path.node_ids)
            if key not in unique or path.score > unique[key].score:
                unique[key] = path
        paths = sorted(
            unique.values(),
            key=lambda p: (len(p.covered_requirements), p.score, -p.depth),
            reverse=True,
        )[: self.beam_width]

        promotions: dict[str, dict] = {}
        for path in paths:
            coverage = len(path.covered_requirements) / max(1, len(req_ids))
            for depth, node_id in enumerate(path.node_ids):
                item = by_id.get(node_id)
                if item is None:
                    promotions.setdefault(node_id, {
                        "id": node_id,
                        "path_score": path.score,
                        "path_coverage": coverage,
                        "path_depth": depth,
                        "path_edge_types": list(path.edge_types),
                    })
                    continue
                md = dict(item.get("metadata") or {})
                md.update({
                    "evidence_path": True,
                    "path_depth": min(depth, self.max_hops),
                    "path_coverage": round(coverage, 4),
                    "path_edge_types": list(path.edge_types),
                })
                item["metadata"] = md
                item["path_score"] = max(float(item.get("path_score", 0.0) or 0.0), path.score)
                item["score"] = float(item.get("score", 0.0) or 0.0) + min(
                    0.12, 0.04 * len(path.covered_requirements) + 0.02 * path.depth
                )
                promotions[node_id] = item

        return paths, promotions
