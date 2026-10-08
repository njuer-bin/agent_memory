from __future__ import annotations

import re
from collections import defaultdict


class MemoryLinkBuilder:
    """Persist lightweight evidence links without turning links into the main retriever.

    Links are provenance-aware edges between raw messages. They are created only
    from explicit causal language or strong same-session continuation signals.
    """

    CAUSAL = (
        (re.compile(r"(?:因为|由于)\s*(.{1,100}?)\s*(?:，|,)??\s*(?:所以|因此|于是)\s*(.{1,100})"), "cause_to_effect"),
        (re.compile(r"(.{1,80}?)\s*(?:导致|造成|引发|使得)\s*(.{1,80})"), "cause_to_effect"),
        (re.compile(r"(.{1,80}?)\s*(?:是因为|源于|的原因是)\s*(.{1,80})"), "effect_to_cause"),
    )
    TEMPORAL = ("后来", "之后", "然后", "接着", "最终", "随后", "第二天", "第二周", "确认", "验证", "有效", "解决")
    UPDATE = ("改成", "更正", "取消", "推迟", "改期", "现在是", "已经", "不再", "换成")

    @classmethod
    def _entities(cls, text: str) -> set[str]:
        """Extract stable lexical anchors for lightweight cross-message linking.

        Chinese does not have whitespace-delimited tokens, so treating a whole
        Chinese span as one entity misses obvious anchors such as 北京 in
        我在北京工作 vs 北京项目后来确认有效. Keep Latin tokens as-is and add
        short Chinese 2-4 character n-grams; short stopwords are removed.
        """
        text = text or ""
        words = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,39}", text))
        chinese_spans = re.findall(r"[\u4e00-\u9fff]+", text)
        stop = {"因为", "所以", "导致", "然后", "现在", "这个", "那个", "后来", "之后", "确认", "有效"}
        for span in chinese_spans:
            if 2 <= len(span) <= 12:
                words.add(span)
            for n in (2, 3, 4):
                for i in range(0, max(0, len(span) - n + 1)):
                    gram = span[i:i + n]
                    if gram not in stop:
                        words.add(gram)
        return {x for x in words if len(x) >= 2}
    @classmethod
    def build(cls, current: dict, previous: list[dict]) -> list[dict]:
        links = []
        content = str(current.get("content") or "")
        current_id = current.get("id")
        session_id = current.get("session_id")
        current_entities = cls._entities(content)

        for pattern, direction in cls.CAUSAL:
            for match in pattern.finditer(content):
                left, right = match.group(1).strip(), match.group(2).strip()
                source_id, target_id = (current_id, current_id)
                # A single message contains both endpoints; keep a self-contained
                # causal link only as an explanatory link, not a traversal edge.
                links.append({
                    "source_id": current_id,
                    "target_id": current_id,
                    "relation": "causal_internal",
                    "confidence": 0.96,
                    "trigger": direction,
                    "evidence": f"{left} -> {right}",
                    "session_id": session_id,
                    "created_at": int(current.get("timestamp") or 0),
                })

        for prev in previous:
            if prev.get("session_id") != session_id:
                continue
            prev_content = str(prev.get("content") or "")
            prev_entities = cls._entities(prev_content)
            overlap = len(current_entities & prev_entities)
            temporal = any(x in content for x in cls.TEMPORAL)
            prev_core = re.sub(r"[。！？!?；;，,、\s]+", "", prev_content)
            current_core = re.sub(r"[。！？!?；;，,、\s]+", "", content)
            lexical_bridge = (
                len(prev_core) >= 5 and prev_core in current_core
            ) or (
                len(current_core) >= 5 and current_core in prev_core
            )
            # Strong continuation: an explicit continuation cue, a shared
            # salient entity, or direct lexical carry-over. The latter is
            # critical for causal chains such as "migration unfinished" ->
            # "because migration unfinished, launch postponed".
            if lexical_bridge:
                links.append({
                    "source_id": prev["id"],
                    "target_id": current_id,
                    "relation": "evidence_continuation",
                    "confidence": 0.94,
                    "trigger": "lexical_carryover",
                    "evidence": "",
                    "session_id": session_id,
                    "created_at": int(current.get("timestamp") or 0),
                })
            elif overlap >= 1 and temporal:
                links.append({
                    "source_id": prev["id"],
                    "target_id": current_id,
                    "relation": "session_continuation",
                    "confidence": min(0.92, 0.68 + 0.08 * overlap),
                    "trigger": "shared_entity_temporal",
                    "evidence": "",
                    "session_id": session_id,
                    "created_at": int(current.get("timestamp") or 0),
                })
            elif overlap >= 2:
                links.append({
                    "source_id": prev["id"],
                    "target_id": current_id,
                    "relation": "topic_continuation",
                    "confidence": min(0.86, 0.62 + 0.07 * overlap),
                    "trigger": "shared_entities",
                    "evidence": "",
                    "session_id": session_id,
                    "created_at": int(current.get("timestamp") or 0),
                })

        # De-duplicate while preserving the strongest trigger.
        best = {}
        for link in links:
            key = (link["source_id"], link["target_id"], link["relation"])
            if key not in best or link["confidence"] > best[key]["confidence"]:
                best[key] = link
        return list(best.values())[:16]
