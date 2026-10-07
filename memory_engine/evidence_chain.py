from __future__ import annotations

import re
from collections import defaultdict, deque


class EvidenceChainBuilder:
    """Build a small query-time evidence graph from ranked memory candidates.

    The goal is not to generate an answer. It identifies connected memory
    fragments that jointly explain a multi-hop query and annotates those
    original memories so final Top-K selection can preserve the bridge.
    """

    EDGE_PATTERNS = (
        re.compile(r"^(.{1,40}?)总部在(.{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:在|位于)(.{1,40}?)(?:工作|上班)$"),
        re.compile(r"^(.{1,40}?)(?:工作于|就职于)(.{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:是|叫|为)([A-Za-z0-9_\u4e00-\u9fff]{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:属于|隶属于|来自)(.{1,40})$"),
    )

    STOP = {
        "用户", "我", "我的", "这个", "那个",
        "什么", "哪里", "哪个", "哪些", "怎么", "如何", "之前", "以前",
    }

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops

    @classmethod
    def _clean_entity(cls, value: str) -> str:
        value = re.sub(r"^[\s，。,:：；;、]+|[\s，。,:：；;、]+$", "", value)
        value = re.sub(r"^(?:我的|我|用户的)", "", value)
        return value.strip()

    @classmethod
    def _edges(cls, item: dict) -> list[tuple[str, str]]:
        md = item.get("metadata") or {}
        subject = cls._clean_entity(str(md.get("subject") or ""))
        obj = cls._clean_entity(str(md.get("object") or md.get("value") or ""))
        edges: list[tuple[str, str]] = []
        if subject and obj and subject not in cls.STOP and obj not in cls.STOP:
            edges.append((subject, obj))

        content = str(item.get("content") or "").strip()
        for pattern in cls.EDGE_PATTERNS:
            for match in pattern.finditer(content):
                left = cls._clean_entity(match.group(1))
                right = cls._clean_entity(match.group(2))
                if left and right and left not in cls.STOP and right not in cls.STOP:
                    edges.append((left, right))

        result = []
        for edge in edges:
            if edge not in result and edge[0] != edge[1]:
                result.append(edge)
        return result[:4]

    def annotate(self, query: str, candidates: list[dict]) -> list[dict]:
        if len(candidates) < 2:
            return candidates

        edges_by_id = {}
        adjacency = defaultdict(list)
        for item in candidates:
            edges = self._edges(item)
            if not edges:
                continue
            edges_by_id[item["id"]] = edges
            for left, right in edges:
                adjacency[left].append((right, item["id"]))
                adjacency[right].append((left, item["id"]))

        if not edges_by_id:
            out = []
            for item in candidates:
                result = dict(item)
                metadata = dict(result.get("metadata") or {})
                metadata["evidence_chain"] = False
                metadata["path_completeness"] = 0.0
                metadata["chain_score"] = 0.0
                result["metadata"] = metadata
                out.append(result)
            return out

        # Build connected components from the retrieved relation edges. A
        # component with >=2 memories is an evidence chain; max_hops is used
        # below to keep the chain bounded when measuring path completeness.
        seed_ids = list(edges_by_id)[:8]
        connected_ids: set[str] = set()

        for seed_id in seed_ids:
            seed_edges = edges_by_id.get(seed_id, [])
            for left, right in seed_edges:
                # Start from both endpoints. An undirected evidence edge must
                # be traversable from either side; starting from only ``left``
                # would miss a chain when the bridge continues through ``right``.
                queue = deque([(left, 0), (right, 0)])
                seen_nodes = {left, right}
                component_items = {seed_id}
                while queue:
                    node, hop = queue.popleft()
                    if hop >= self.max_hops:
                        continue
                    for other, item_id in adjacency.get(node, []):
                        component_items.add(item_id)
                        if other not in seen_nodes:
                            seen_nodes.add(other)
                            queue.append((other, hop + 1))
                if len(component_items) >= 2:
                    connected_ids.update(component_items)

        # A connected chain is evidence only when it spans at least two
        # memory fragments. Score is deliberately small: retrieval/reranking
        # remains primary, while chain completeness breaks bridge ties.
        out = []
        for item in candidates:
            result = dict(item)
            metadata = dict(result.get("metadata") or {})
            if item["id"] in connected_ids:
                chain_score = min(0.12, 0.04 * max(1, len(edges_by_id.get(item["id"], []))))
                metadata["evidence_chain"] = True
                metadata["path_completeness"] = 1.0
                metadata["chain_score"] = chain_score
                result["score"] = round(float(result.get("score", 0.0)) + chain_score, 6)
            else:
                metadata["evidence_chain"] = False
                metadata["path_completeness"] = 0.0
                metadata["chain_score"] = 0.0
            result["metadata"] = metadata
            out.append(result)

        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out
