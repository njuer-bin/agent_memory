from __future__ import annotations

from collections import OrderedDict


class EvidenceBuilder:
    def build(self, candidates, top_k):
        """
        Evidence completeness：
        同一事实的 raw + structured evidence 不重复刷屏；
        优先保留不同 memory_type 的互补证据。
        """
        selected = []
        seen_content = set()
        type_count = {}

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

        # P2：邻近上下文承担“证据补全”职责。reranker 只负责排序，
        # 不能因为邻居与问题的字面重合较少，就让关键上下文全部掉出 Top-K。
        # 在不超过 top_k 的前提下，最多补入 2 条 context-expanded 原始消息。
        context_items = [
            item for item in candidates
            if (item.get("metadata") or {}).get("context_expanded")
            and item["content"].strip() not in seen_content
        ]
        for item in context_items[:2]:
            if len(selected) < top_k:
                selected.append(item)
                seen_content.add(item["content"].strip())
                continue

            # 已满时，用最低分的非上下文证据替换；绝不互相替换上下文。
            replace_idx = next(
                (
                    i for i in range(len(selected) - 1, -1, -1)
                    if not (selected[i].get("metadata") or {}).get("context_expanded")
                ),
                None,
            )
            if replace_idx is not None:
                selected[replace_idx] = item
                seen_content.add(item["content"].strip())

        return selected
