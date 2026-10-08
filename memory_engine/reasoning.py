from __future__ import annotations

import re
from collections import defaultdict


class MemoryReasoner:
    """Deterministic query-time reasoning layer for AML reasoning gaps."""

    STATE_MARKERS = (
        "现在", "目前", "当前", "最新", "现居", "如今", "current",
        "currently", "now", "latest", "present", "today",
    )
    CAUSAL_MARKERS = (
        "为什么", "为何", "原因", "导致", "因为", "所以", "因此", "结果",
        "why", "cause", "caused", "because", "therefore", "led to", "result",
    )
    CAUSAL_PREDICATES = {
        "causes", "leads_to", "caused_by", "because_of", "results_in",
        "resulted_in", "triggered_by", "reason",
    }
    STOP = {
        "what", "which", "who", "where", "when", "why", "how", "many", "much",
        "does", "did", "do", "has", "have", "had", "is", "are", "was", "were",
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at",
        "with", "from", "by", "both", "common", "shared", "current", "now",
        "currently", "latest", "present", "result", "cause", "because",
    }

    @staticmethod
    def _tokens(text: str) -> set[str]:
        raw = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|[\u4e00-\u9fff]{2,}|\d{4}", text or "")
        return {x.lower() for x in raw if x.lower() not in MemoryReasoner.STOP}

    @classmethod
    def _overlap(cls, query: str, text: str) -> float:
        q = cls._tokens(query)
        if not q:
            return 0.0
        t = cls._tokens(text)
        return len(q & t) / max(1, len(q))

    @classmethod
    def _is_state_query(cls, query: str) -> bool:
        q = query.lower()
        return any(marker in q for marker in cls.STATE_MARKERS)

    @classmethod
    def _is_causal_query(cls, query: str) -> bool:
        q = query.lower()
        return any(marker in q for marker in cls.CAUSAL_MARKERS)

    @staticmethod
    def _state_score(query: str, fact: dict, predicate_hint: str | None) -> float:
        md_text = " ".join(
            str(fact.get(k) or "")
            for k in ("subject", "predicate", "object", "content")
        )
        overlap = MemoryReasoner._overlap(query, md_text)
        predicate = str(fact.get("predicate") or "")
        score = 0.80 + min(0.12, overlap * 0.12)
        if predicate_hint and predicate == predicate_hint:
            score += 0.10
        if fact.get("status") == "active":
            score += 0.04
        return min(1.25, score)

    @classmethod
    def _temporal_score(cls, query: str, item: dict) -> float:
        text = " ".join(
            str(item.get(k) or "")
            for k in ("content", "valid_from", "valid_to")
        )
        q_years = set(re.findall(r"(?:19|20)\d{2}", query or ""))
        q_months = {
            x.lower() for x in re.findall(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
                query or "", flags=re.I,
            )
        }
        score = 0.0
        if q_years and q_years.intersection(re.findall(r"(?:19|20)\d{2}", text)):
            score += 0.12
        if q_months and q_months.intersection({x.lower() for x in re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            text, flags=re.I,
        )}):
            score += 0.08
        if score:
            return score
        markers = re.findall(r"\d{4}年\d{1,2}月|\d{4}年|去年|前年|今年|上个月|本月|之前|以前|后来|之后", query or "")
        return min(0.12, 0.04 * sum(1 for m in markers if m in text))

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[\s，。！？、,.!?；;:：]+", "", str(text or "")).lower()

    @classmethod
    def _protect_current_state(cls, query: str, out: list[dict], store, user_id: str):
        """Promote active facts and suppress stale evidence for state queries.

        P6 intentionally keeps historical raw/window evidence for recall. For a
        current-state question, however, returning a stale fact in the first few
        slots is harmful: the answer model can treat the old value as current.
        We therefore keep historical rows available, but give rows containing a
        superseded fact a strong negative score. This also handles derived
        windows that contain both the old and new turn.
        """
        active = store.active_facts(user_id, include_history=False)
        history = store.active_facts(user_id, include_history=True)
        active_ids = {str(f.get("id")) for f in active}
        stale = [f for f in history if str(f.get("id")) not in active_ids and f.get("status") != "active"]

        stale_texts = []
        for fact in stale:
            content = cls._normalize(fact.get("content", ""))
            if content:
                stale_texts.append((content, fact.get("supersedes_id")))

        for item in out:
            content = cls._normalize(item.get("content", ""))
            stale_hit = False
            for old_text, _ in stale_texts:
                if old_text and old_text in content:
                    stale_hit = True
                    break
            if not stale_hit:
                continue
            item["score"] = min(float(item.get("score", 0.0)), -0.20)
            md = dict(item.get("metadata") or {})
            md["state_reasoning"] = True
            md["stale_state_evidence"] = True
            item["metadata"] = md

    @classmethod
    def augment(cls, query: str, plan, ranked: list[dict], store, user_id: str) -> list[dict]:
        out = [dict(item) for item in ranked]
        by_id = {item.get("id") for item in out}

        if cls._is_state_query(query):
            for fact in store.active_facts(user_id, include_history=False):
                score = cls._state_score(query, fact, getattr(plan, "predicate_hint", None))
                if score < 0.80 or fact.get("id") in by_id:
                    continue
                out.append({
                    "id": fact["id"],
                    "content": fact["content"],
                    "role": "memory",
                    "timestamp": fact["timestamp"],
                    "user_id": user_id,
                    "session_id": "",
                    "score": round(score, 6),
                    "source": "state_reasoning",
                    "memory_type": "fact",
                    "status": fact.get("status", "active"),
                    "valid_from": fact.get("valid_from", fact["timestamp"]),
                    "valid_to": fact.get("valid_to"),
                    "metadata": {
                        "subject": fact.get("subject"),
                        "predicate": fact.get("predicate"),
                        "object": fact.get("object"),
                        "supersedes_id": fact.get("supersedes_id"),
                        "state_reasoning": True,
                        "current_state": True,
                    },
                })
                by_id.add(fact["id"])

            # Apply stale suppression after active facts have been injected.
            cls._protect_current_state(query, out, store, user_id)

        if getattr(plan, "temporal", False):
            for item in out:
                bonus = cls._temporal_score(query, item)
                if bonus:
                    item["score"] = round(float(item.get("score", 0.0)) + bonus, 6)
                    md = dict(item.get("metadata") or {})
                    md["temporal_reasoning"] = True
                    item["metadata"] = md

        if cls._is_causal_query(query):
            relations = store.relations(user_id)
            query_entities = cls._tokens(query)
            causal = []
            for rel in relations:
                predicate = str(rel.get("predicate") or "")
                content = str(rel.get("content") or "")
                if predicate not in cls.CAUSAL_PREDICATES and not re.search(
                    r"因为|由于|导致|造成|引发|使得|是因为|源于|because|caused|led to|resulted",
                    content, flags=re.I,
                ):
                    continue
                endpoint_text = " ".join(
                    str(rel.get(k) or "") for k in ("subject", "object", "content")
                )
                overlap = cls._overlap(query, endpoint_text)
                entity_hit = any(str(rel.get(k) or "").lower() in query_entities for k in ("subject", "object"))
                causal.append((
                    0.92 + (0.08 if entity_hit else 0.0) + min(0.06, overlap * 0.06),
                    rel,
                ))
            causal.sort(key=lambda x: x[0], reverse=True)
            for score, rel in causal[:12]:
                rid = rel.get("id")
                if rid in by_id:
                    for item in out:
                        if item.get("id") == rid:
                            item["score"] = max(float(item.get("score", 0.0)), score)
                            md = dict(item.get("metadata") or {})
                            md["causal_reasoning"] = True
                            md["causal_edge"] = True
                            item["metadata"] = md
                            break
                    continue
                out.append({
                    "id": rid,
                    "content": rel["content"],
                    "role": "relation",
                    "timestamp": rel["timestamp"],
                    "user_id": user_id,
                    "session_id": "",
                    "score": round(score, 6),
                    "source": "causal_reasoning",
                    "memory_type": "relation",
                    "status": "active",
                    "valid_from": rel["timestamp"],
                    "valid_to": None,
                    "metadata": {
                        "subject": rel.get("subject"),
                        "predicate": predicate,
                        "object": rel.get("object"),
                        "causal_reasoning": True,
                        "causal_edge": True,
                    },
                })
                by_id.add(rid)

            raws = store.all_raw(user_id)
            for raw in raws:
                content = str(raw.get("content") or "")
                if not re.search(r"因为|由于|导致|造成|引发|使得|是因为|源于|because|caused|led to|resulted", content, flags=re.I):
                    continue
                if raw["id"] in by_id:
                    continue
                overlap = cls._overlap(query, content)
                if overlap <= 0 and len(out) > 100:
                    continue
                out.append({
                    "id": raw["id"],
                    "content": content,
                    "role": raw["role"],
                    "timestamp": raw["timestamp"],
                    "user_id": user_id,
                    "session_id": raw["session_id"],
                    "score": round(0.86 + min(0.06, overlap * 0.06), 6),
                    "source": "causal_provenance",
                    "memory_type": "raw",
                    "status": "active",
                    "valid_from": raw["timestamp"],
                    "valid_to": None,
                    "metadata": {
                        "source_message_ids": [raw["id"]],
                        "causal_reasoning": True,
                    },
                })
                by_id.add(raw["id"])

        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out
