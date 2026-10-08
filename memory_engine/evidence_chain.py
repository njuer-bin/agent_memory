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
        re.compile(r"^(.{1,40}?)总部\s*(?:在|位于)\s*(.{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:在|位于)\s*(.{1,40}?)(?:工作|上班)$"),
        re.compile(r"^(.{1,40}?)(?:工作于|就职于)\s*(.{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:是|叫|为)\s*(.{1,40})$"),
        re.compile(r"^(.{1,40}?)(?:属于|隶属于|来自)\s*(.{1,40})$"),
    )

    # Directed causal edges are kept separate from generic entity edges.
    # They let a multi-hop query preserve cause -> intermediate -> outcome
    # evidence instead of treating every connected entity as equivalent.
    CAUSAL_PATTERNS = (
        re.compile(r"(?:因为|由于)\s*(.{1,80}?)\s*(?:，|,)??\s*(?:所以|因此|于是)\s*(.{1,80})"),
        re.compile(r"(.{1,60}?)\s*(?:导致|造成|引发|使得)\s*(.{1,60})"),
        re.compile(r"(.{1,60}?)\s*(?:是因为|源于)\s*(.{1,60})"),
        re.compile(r"(.{1,60}?)\s*(?:的原因是)\s*(.{1,60})"),
    )

    FRIEND_PATTERNS = (
        re.compile(r"^(?:我的|我|用户的)?(?:朋友|同事|老板)\s*(?:是|叫|为)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,40})$"),
        re.compile(r"^(?:我的|我|用户的)?(?:朋友|同事|老板)\s+([A-Za-z0-9_\u4e00-\u9fff]{1,40})$"),
        re.compile(r"^([A-Za-z0-9_\u4e00-\u9fff]{1,40})\s*(?:是|叫|为)\s*(?:我的|我|用户的)?(?:朋友|同事|老板)$"),
    )

    STOP = {
        "用户", "我", "我的", "这个", "那个",
        "什么", "哪里", "哪个", "哪些", "怎么", "如何", "之前", "以前",
    }

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops

    @classmethod
    def _clean_entity(cls, value: str) -> str:
        value = re.sub(r"^[\s，。,:：；;、!?！？…]+|[\s，。,:：；;、!?！？…]+$", "", value)
        value = re.sub(r"^(?:我的|我|用户的)", "", value)
        value = re.sub(r"^(?:那个|之前提到的|之前说的)\s*", "", value)
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
        # Normalize common conversational punctuation first, then inspect each
        # clause independently. This keeps extraction tolerant without making
        # the graph depend on one exact sentence template.
        clauses = [
            cls._clean_entity(part)
            for part in re.split(r"[。！？!?；;，,、\n]+", content)
            if cls._clean_entity(part)
        ]
        if not clauses:
            clauses = [cls._clean_entity(content)]

        for clause in clauses:
            for pattern in cls.EDGE_PATTERNS:
                for match in pattern.finditer(clause):
                    left = cls._clean_entity(match.group(1))
                    right = cls._clean_entity(match.group(2))
                    if left and right and left not in cls.STOP and right not in cls.STOP:
                        edges.append((left, right))

            # Preserve causal direction.  For "effect 是因为 cause" and
            # "effect 的原因是 cause", reverse the surface order so traversal
            # remains cause -> effect.
            for index, pattern in enumerate(cls.CAUSAL_PATTERNS):
                for match in pattern.finditer(clause):
                    left = cls._clean_entity(match.group(1))
                    right = cls._clean_entity(match.group(2))
                    if not left or not right:
                        continue
                    cause, effect = (left, right) if index < 2 else (right, left)
                    if cause not in cls.STOP and effect not in cls.STOP:
                        edges.append((cause, effect))

            for pattern in cls.FRIEND_PATTERNS:
                for match in pattern.finditer(clause):
                    right = cls._clean_entity(match.group(1))
                    if right and right not in cls.STOP:
                        edges.append(("朋友", right))

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
                chain_size = len(connected_ids)
                chain_score = min(0.16, 0.025 * max(1, chain_size))
                metadata["evidence_chain"] = True
                metadata["path_completeness"] = min(1.0, chain_size / 3.0)
                metadata["causal_chain"] = bool(edges_by_id.get(item["id"], []))
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
