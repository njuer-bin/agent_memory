from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidencePath:
    """A bounded query-time path over already retrieved/provenance evidence."""

    node_ids: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    covered_requirements: set[str] = field(default_factory=set)
    score: float = 0.0

    @property
    def depth(self) -> int:
        return max(0, len(self.node_ids) - 1)


class EvidencePathReconstructor:
    """Reconstruct small evidence paths instead of expanding the whole graph.

    The retriever remains the source of candidates. This layer only follows
    high-confidence persisted links and structured relations around those
    candidates, using a small beam. Raw messages are fetched for promoted
    path nodes so benchmark evidence is never represented only by a derived
    relation.
    """

    def __init__(self, max_hops: int = 3, beam_width: int = 8, neighbors_per_node: int = 6):
        self.max_hops = max(1, max_hops)
        self.beam_width = max(1, beam_width)
        self.neighbors_per_node = max(1, neighbors_per_node)

    @staticmethod
    def _requirement_score(item: dict[str, Any]) -> float:
        return float(item.get("_requirement_score", 0.0) or 0.0)

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

        # Structured relations connect derived memories. Keep their relation
        # semantics so causal/final-hop traversal can be preferred later.
        for rel in relation_rows:
            rid = str(rel.get("id") or "")
            if not rid:
                continue
            confidence = 0.90 if rel.get("predicate") in {"causes", "headquarters", "belongs_to", "located_in"} else 0.72
            subject_id = str(rel.get("subject_id") or "")
            object_id = str(rel.get("object_id") or "")
            if subject_id and object_id:
                graph[subject_id].append((object_id, str(rel.get("predicate") or "relation"), confidence))
                graph[object_id].append((subject_id, str(rel.get("predicate") or "relation"), confidence))
            # Relation IDs can still act as a bridge when endpoint IDs are not
            # persisted in older databases; connect through textual endpoints
            # in candidate metadata in the engine rather than inventing IDs.
            if rid and subject_id:
                graph[rid].append((subject_id, str(rel.get("predicate") or "relation"), confidence))
                graph[subject_id].append((rid, str(rel.get("predicate") or "relation"), confidence))

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

        # Seed from direct retrieval, not from arbitrary graph nodes. This is
        # the main guard against graph noise and path explosion.
        seeds = sorted(
            by_id.values(),
            key=lambda x: (
                len(set(x.get("_evidence_requirements", []) or [])),
                self._requirement_score(x),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )[:16]

        paths: list[EvidencePath] = []
        for seed in seeds:
            sid = str(seed.get("id"))
            covered = set(seed.get("_evidence_requirements", []) or []) & req_ids
            initial = EvidencePath([sid], [], covered, float(seed.get("score", 0.0) or 0.0))
            beam = [initial]
            for _ in range(self.max_hops):
                expanded: list[EvidencePath] = []
                for path in beam:
                    node = path.node_ids[-1]
                    for other, edge_type, confidence in graph.get(node, [])[: self.neighbors_per_node]:
                        if other in path.node_ids:
                            continue
                        item = by_id.get(other)
                        # Only follow graph nodes that are already retrieved or
                        # whose raw provenance can be recovered by the engine.
                        if item is None:
                            continue
                        new_cov = set(path.covered_requirements)
                        new_cov.update(item.get("_evidence_requirements", []) or [])
                        path_score = (
                            0.45 * float(item.get("score", 0.0) or 0.0)
                            + 0.35 * (len(new_cov) / max(1, len(req_ids)))
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

        # Keep only useful paths, deduplicate exact node sequences, and prefer
        # coverage over raw similarity. This is deliberately small and bounded.
        unique = {}
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
            for depth, node_id in enumerate(path.node_ids):
                item = by_id.get(node_id)
                if not item:
                    continue
                md = dict(item.get("metadata") or {})
                md["evidence_path"] = True
                md["path_depth"] = min(depth, self.max_hops)
                md["path_coverage"] = round(len(path.covered_requirements) / max(1, len(req_ids)), 4)
                md["path_edge_types"] = list(path.edge_types)
                item["metadata"] = md
                item["path_score"] = max(float(item.get("path_score", 0.0) or 0.0), path.score)
                item["score"] = float(item.get("score", 0.0) or 0.0) + min(0.12, 0.04 * len(path.covered_requirements) + 0.02 * path.depth)
                promotions[node_id] = item

        return paths, promotions
