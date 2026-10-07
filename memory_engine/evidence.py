from __future__ import annotations

from collections import OrderedDict


class EvidenceBuilder:
    def build(self, candidates, top_k, multi_hop=False):
        """
        Evidence completeness：
        同一事实的 raw + structured evidence 不重复刷屏；
        优先保留不同 memory_type 的互补证据。
        """
        selected = []
        seen_content = set()
        type_count = {}

        if multi_hop:
            # Keep connected bridge evidence from being discarded by the final
            # diversity pass. This is a bounded ranking bonus, not a new source.
            anchors = set()
            paths = []
            for item in candidates:
                md = item.get("metadata") or {}
                anchors.update(md.get("evidence_anchors", []) or [])
                path = tuple(md.get("graph_path", []) or [])
                if path:
                    paths.append(path)
            for item in candidates:
                md = item.get("metadata") or {}
                item_anchors = set(md.get("evidence_anchors", []) or [])
                path = tuple(md.get("graph_path", []) or [])
                connected = bool(item_anchors & anchors)
                if path:
                    connected = connected or any(
                        set(path) & set(other)
                        for other in paths
                        if other != path
                    )
                if connected:
                    item["score"] = float(item.get("score", 0.0)) + 0.04
            candidates = sorted(
                candidates,
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )

        # 第一轮：保证证据类型多样性
        for item in candidates:
            content = item["content"].strip()
            if not content or content in seen_content:
                continue
            mt = item.get("memory_type","raw")
            if type_count.get(mt,0) >= 3:
                continue
            selected.append(item)
            seen_content.add(content)
            type_count[mt] = type_count.get(mt,0) + 1
            if len(selected) >= top_k:
                break

        return selected
