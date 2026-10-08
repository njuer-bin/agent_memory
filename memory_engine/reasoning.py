from __future__ import annotations

import re
from collections import defaultdict


class MemoryReasoner:
    """Deterministic query-time reasoning layer for AML reasoning gaps.

    This is deliberately not a graph database or a replacement for P6 hybrid
    retrieval. It operates on the already stored facts/events/relations and
    protects three capabilities that retrieval alone does not express well:
    current-state selection, temporal evidence ordering, and causal evidence
    recall.
    """

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
        # Active facts are the governed current state; history is intentionally
        # never promoted by this path.
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
        # Chinese absolute/relative temporal wording often survives only in
        # the original message; preserve a small lexical bonus.
        markers = re.findall(r"\d{4}年\d{1,2}月|\d{4}年|去年|前年|今年|上个月|本月|之前|以前|后来|之后", query or "")
        return min(0.12, 0.04 * sum(1 for m in markers if m in text))

    @classmethod
    def augment(cls, query: str, plan, ranked: list[dict], store, user_id: str) -> list[dict]:
        out = [dict(item) for item in ranked]
        by_id = {item.get("id") for item in out}

        # D1: current-state protection.  The governance layer already marks
        # superseded facts inactive; this query-time layer makes that state
        # visible even when lexical/dense retrieval preferred the old raw turn.
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

        # C1: temporal evidence ordering.  Do not hard-filter here: a date
        # mention can be attached to a neighboring raw message while the event
        # itself has a different timestamp. A positive score is safer than
        # deleting potentially necessary evidence.
        if getattr(plan, "temporal", False):
            for item in out:
                bonus = cls._temporal_score(query, item)
                if bonus:
                    item["score"] = round(float(item.get("score", 0.0)) + bonus, 6)
                    md = dict(item.get("metadata") or {})
                    md["temporal_reasoning"] = True
                    item["metadata"] = md

        # B2: causal evidence recall.  The analyzer already extracts directed
        # `causes` relations, but ordinary retrieval can discard them because
        # causal questions contain abstract words rather than the exact cause
        # text. Promote causal edges independently, then EvidenceChainBuilder
        # can connect them with the rest of the candidate set.
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

            # Also retain the raw causal sentence(s), because AML evidence is
            # ultimately message-level and the directed relation alone may not
            # contain enough linguistic context for the answer model.
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
