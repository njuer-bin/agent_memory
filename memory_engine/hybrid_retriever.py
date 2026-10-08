from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

from .store import BM25
from .vector import cosine
from .vector_index import MemoryVectorIndex


class HybridRetriever:
    """Evidence-completeness-first hybrid retriever.

    P6 deliberately keeps raw messages as the primary evidence source and
    derives additional retrieval views without deleting or replacing them:
    single-message, adjacent-message windows, and session segments.  Every
    derived view carries source_message_ids so the answer layer can trace it
    back to the original messages.
    """

    WINDOW_RADIUS = 2
    WINDOW_MAX_CHARS = 2400
    SESSION_MAX_MESSAGES = 12

    def __init__(self, store, embedder, vector_index: MemoryVectorIndex | None = None):
        self.store = store
        self.embedder = embedder
        self.vector_index = vector_index or MemoryVectorIndex(store)

    @staticmethod
    def _window_id(source_ids: list[str], view_type: str = "window") -> str:
        # Window and session views may contain the same source IDs (e.g. a
        # three-message session). Keep their IDs distinct so one view cannot
        # overwrite the other in the fusion map.
        digest = hashlib.sha1(
            f"{view_type}|{'|'.join(source_ids)}".encode("utf-8")
        ).hexdigest()[:20]
        return f"{view_type}_{digest}"

    def _raw_views(self, raws: list[dict], user_id: str, query: str = "") -> list[dict]:
        """Build raw + local-window views while preserving source IDs."""
        docs = []
        # Raw message view: never summarize or replace the original message.
        for idx, r in enumerate(raws):
            docs.append({
                "id": r["id"], "content": r["content"], "role": r["role"],
                "timestamp": r["timestamp"], "user_id": user_id,
                "session_id": r["session_id"], "memory_type": "raw",
                "status": "active", "source": "raw",
                "valid_from": r["timestamp"], "valid_to": None,
                "metadata": {
                    "request_id": r["request_id"],
                    "source_message_ids": [r["id"]],
                    "view": "message",
                    "message_index": idx,
                },
            })

        # Adjacent windows are intentionally local to one session.  This is
        # the primary P6 fix for multi-hop evidence split across neighboring
        # turns: if one message is retrieved, its nearby turns travel with it.
        by_session: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(raws, key=lambda x: (x["session_id"], x["timestamp"])):
            by_session[r["session_id"]].append(r)

        # Causal/path questions need more local context than ordinary fact
        # lookup: the trigger, transition, and outcome are often adjacent
        # turns. Keep the default window small, but expand it deterministically
        # for causal markers. This is retrieval-only context expansion; raw
        # messages remain the canonical evidence.
        causal_query = any(
            marker in (query or "")
            for marker in ("为什么", "为何", "原因", "导致", "因为", "所以", "因此",
                           "how did", "why", "cause", "caused", "because")
        )
        window_radius = 4 if causal_query else self.WINDOW_RADIUS

        for session_id, session_rows in by_session.items():
            for idx, center in enumerate(session_rows):
                lo = max(0, idx - window_radius)
                hi = min(len(session_rows), idx + window_radius + 1)
                members = session_rows[lo:hi]
                source_ids = [m["id"] for m in members]
                content = "\n".join(str(m["content"]) for m in members)
                if len(content) > self.WINDOW_MAX_CHARS:
                    content = content[: self.WINDOW_MAX_CHARS]
                if not content.strip():
                    continue
                docs.append({
                    "id": self._window_id(source_ids, "window"),
                    "content": content,
                    "role": "context",
                    "timestamp": center["timestamp"],
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": "window",
                    "status": "active",
                    "source": "session_window",
                    "valid_from": members[0]["timestamp"],
                    "valid_to": members[-1]["timestamp"],
                    "metadata": {
                        "source_message_ids": source_ids,
                        "view": "window",
                        "center_message_id": center["id"],
                        "session_id": session_id,
                    },
                })

            # A bounded session segment gives the retriever a coarser view for
            # questions whose evidence spans several turns. It is still fully
            # traceable to the original message IDs.
            for start in range(0, len(session_rows), self.SESSION_MAX_MESSAGES):
                members = session_rows[start:start + self.SESSION_MAX_MESSAGES]
                if len(members) < 2:
                    continue
                source_ids = [m["id"] for m in members]
                content = "\n".join(str(m["content"]) for m in members)
                if len(content) > self.WINDOW_MAX_CHARS * 2:
                    content = content[: self.WINDOW_MAX_CHARS * 2]
                docs.append({
                    "id": self._window_id(source_ids, "session"),
                    "content": content,
                    "role": "context",
                    "timestamp": members[-1]["timestamp"],
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": "session",
                    "status": "active",
                    "source": "session_segment",
                    "valid_from": members[0]["timestamp"],
                    "valid_to": members[-1]["timestamp"],
                    "metadata": {
                        "source_message_ids": source_ids,
                        "view": "session",
                        "session_id": session_id,
                    },
                })
        return docs

    @staticmethod
    def _normalize_source_text(value: str) -> str:
        return " ".join(str(value or "").lower().split())

    def _attach_source_message_ids(self, docs: list[dict], raws: list[dict]) -> None:
        """Attach raw-message provenance to structured memories."""
        if not raws:
            return
        raw_pairs = [
            (r["id"], self._normalize_source_text(r.get("content", "")))
            for r in raws
        ]
        for doc in docs:
            if doc.get("memory_type") in {"raw", "window", "session"}:
                continue
            content = self._normalize_source_text(doc.get("content", ""))
            if not content:
                continue
            source_ids = []
            for raw_id, raw_text in raw_pairs:
                if raw_text and (content in raw_text or raw_text in content):
                    source_ids.append(raw_id)
            if source_ids:
                md = doc.setdefault("metadata", {})
                md["source_message_ids"] = list(dict.fromkeys(source_ids[:8]))

    @staticmethod
    def _deterministic_signal(query: str, doc: dict) -> float:
        """Small deterministic signals that complement dense/sparse retrieval."""
        q = (query or "").strip().lower()
        if not q:
            return 0.0
        text = str(doc.get("content") or "").lower()
        md = doc.get("metadata", {}) or {}
        signal = 0.0
        # Exact query substring is high-confidence but deliberately capped.
        if len(q) >= 2 and q in text:
            signal += 0.045
        # Structured entity/date/number overlap is robust to embedding drift.
        q_entities = set(__import__("re").findall(r"[A-Za-z][A-Za-z0-9_-]{1,39}|[\u4e00-\u9fff]{2,}", q))
        for key in ("subject", "object", "value", "event"):
            value = str(md.get(key) or "").lower().strip()
            if value and len(value) >= 2 and value in q_entities:
                signal += 0.020
                break
        q_numbers = set(__import__("re").findall(r"\d+(?:\.\d+)?", q))
        if q_numbers and q_numbers.intersection(__import__("re").findall(r"\d+(?:\.\d+)?", text)):
            signal += 0.018
        # Window/session views get a small evidence-preservation bonus. They
        # are not allowed to overwhelm direct message matches.
        if doc.get("memory_type") in {"window", "session"}:
            signal += 0.004
        return signal

    def candidates(self, user_id, query, top_k=30, include_history=False,
                   session_id=None, start_time=None, end_time=None,
                   memory_types=None, memory_type_hint=None,
                   temporal_relation="at", relation_hint=False, sparse_query=None, predicate_hint=None, intent_hint=None,
                   use_dense=True):
        raws = self.store.all_raw(user_id, session_id=session_id)
        if start_time is not None:
            raws = [r for r in raws if r["timestamp"] >= start_time]
        if end_time is not None:
            raws = [r for r in raws if r["timestamp"] <= end_time]

        facts = self.store.active_facts(user_id, include_history=include_history)
        events = self.store.events(user_id, start_time, end_time)
        relations = self.store.relations(user_id)
        rules = self.store.rules(user_id)
        profiles = self.store.profiles(user_id)

        docs = self._raw_views(raws, user_id, query)

        for f in facts:
            docs.append({
                "id": f["id"], "content": f["content"], "role": "memory",
                "timestamp": f["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "fact",
                "status": f["status"], "source": "atomic_fact",
                "valid_from": f["valid_from"], "valid_to": f["valid_to"],
                "metadata": {"subject":f["subject"],"predicate":f["predicate"],
                             "object":f["object"],"supersedes_id":f["supersedes_id"],
                             "source":f.get("source", "user"),
                             "conflict_status":f.get("conflict_status", "none"),
                             "conflict_group_id":f.get("conflict_group_id")},
            })

        for e in events:
            docs.append({
                "id": e["id"], "content": e["content"], "role": "event",
                "timestamp": e["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "event",
                "status": "active", "source": "timeline",
                "valid_from": e.get("event_start") or e["timestamp"],
                "valid_to": e.get("event_end"),
                "metadata": {
                    "event": e["event"],
                    "temporal_text": e.get("temporal_text") or "",
                },
            })

        for r in relations:
            docs.append({
                "id": r["id"], "content": r["content"], "role": "relation",
                "timestamp": r["timestamp"], "user_id": user_id, "session_id": "",
                "memory_type": "relation", "status": "active", "source": "graph",
                "valid_from": r["timestamp"], "valid_to": None,
                "metadata": {"subject":r["subject"],"predicate":r["predicate"],"object":r["object"]},
            })

        for r in rules:
            docs.append({
                "id": r["id"], "content": r["content"], "role": "rule",
                "timestamp": r["timestamp"], "user_id": user_id, "session_id": "",
                "memory_type": "rule", "status": "active", "source": "rule",
                "valid_from": r["timestamp"], "valid_to": None, "metadata": {},
            })

        for p in profiles:
            docs.append({
                "id": f"profile:{p['user_id']}:{p['key']}", "content": p["content"],
                "role": "profile", "timestamp": p["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "profile", "status": "active",
                "source": "profile", "valid_from": p["timestamp"], "valid_to": None,
                "metadata": {"key":p["key"],"value":p["value"]},
            })

        # Attach provenance from structured memories back to the original raw
        # message(s). LoCoMo evaluates evidence at message level, so a
        # structured hit must remain traceable to the message that produced it.
        self._attach_source_message_ids(docs, raws)

        if memory_types:
            allowed = set(memory_types)
            docs = [d for d in docs if d["memory_type"] in allowed]
        if not docs:
            return []

        # Search each retrieval view independently, then fuse ranks. This is
        # intentionally Weighted RRF rather than score mixing: dense and BM25
        # scores are not calibrated to the same scale.
        t_profile = time.perf_counter()
        bm = BM25()
        bm.fit(docs)
        sparse = bm.search(sparse_query or query, top_k=min(max(80, top_k), len(docs)))
        bm25_ms = (time.perf_counter() - t_profile) * 1000
        sparse_rank = {d["id"]: i + 1 for i, (d, _) in enumerate(sparse)}

        embedding_ms = 0.0
        vector_load_ms = 0.0
        dense_rank = {}
        dense_scores = {}
        qv = None
        if use_dense:
            t_profile = time.perf_counter()
            qv = self.embedder.embed(query)
            embedding_ms = (time.perf_counter() - t_profile) * 1000
            t_profile = time.perf_counter()
            # Stored embeddings cover atomic/raw memories. For derived windows,
            # compute a dense representation from their source message vectors.
            stored_ids = [d["id"] for d in docs if d["memory_type"] not in {"window", "session"}]
            dense_pairs = self.vector_index.search(
                user_id, qv, ids=stored_ids, top_k=min(max(80, top_k), len(stored_ids))
            ) if stored_ids else []
            dense_rank = {mid: i + 1 for i, (mid, _) in enumerate(dense_pairs)}
            dense_scores.update({mid: float(score) for mid, score in dense_pairs})
            source_ids = []
            for d in docs:
                if d["memory_type"] in {"window", "session"}:
                    source_ids.extend(d.get("metadata", {}).get("source_message_ids", []))
            vectors = self.store.embeddings_by_ids(user_id, source_ids)
            qdim = len(qv or [])
            derived_view_count = 0
            skipped_bad_dim = 0
            dimension_counts = defaultdict(int)
            for d in docs:
                if d["memory_type"] not in {"window", "session"}:
                    continue
                mids = d.get("metadata", {}).get("source_message_ids", [])
                vals = []
                for mid in mids:
                    vec = vectors.get(mid)
                    if not isinstance(vec, (list, tuple)) or not vec:
                        continue
                    try:
                        vec = [float(x) for x in vec]
                    except (TypeError, ValueError):
                        continue
                    dimension_counts[len(vec)] += 1
                    # Persistent stores can contain vectors produced by an
                    # older embedding model/dimension. Never average vectors
                    # with different dimensions, and prefer vectors matching
                    # the current query embedding.
                    if qdim and len(vec) != qdim:
                        skipped_bad_dim += 1
                        continue
                    vals.append(vec)
                if not vals:
                    continue
                dim = qdim or len(vals[0])
                if any(len(v) != dim for v in vals):
                    skipped_bad_dim += sum(1 for v in vals if len(v) != dim)
                    vals = [v for v in vals if len(v) == dim]
                if not vals:
                    continue
                avg = [sum(v[i] for v in vals) / len(vals) for i in range(dim)]
                score = float(cosine(qv, avg))
                dense_scores[d["id"]] = score
                derived_view_count += 1
            if os.getenv("MEMORY_DIAGNOSTICS", "").strip() == "1":
                logger.info(
                    "HYBRID_DIAGNOSTICS query=%r docs=%d sparse=%d dense=%d derived=%d "
                    "query_dim=%d skipped_dim=%d vector_dims=%s",
                    query, len(docs), len(sparse_rank), len(dense_rank),
                    derived_view_count, qdim, skipped_bad_dim,
                    dict(sorted(dimension_counts.items())),
                )
            derived = sorted(
                ((mid, score) for mid, score in dense_scores.items() if mid not in dense_rank),
                key=lambda x: x[1], reverse=True,
            )[:min(max(80, top_k), len(docs))]
            start_rank = len(dense_rank) + 1
            for offset, (mid, score) in enumerate(derived):
                dense_rank[mid] = start_rank + offset
            vector_load_ms = (time.perf_counter() - t_profile) * 1000

        rrf_k = 60.0
        merged = defaultdict(float)
        # Dense gets slightly more weight because raw-message embeddings are
        # strong for paraphrased evidence; lexical remains independently strong.
        for mid, rank in sparse_rank.items():
            merged[mid] += 1.10 / (rrf_k + rank)
        for mid, rank in dense_rank.items():
            merged[mid] += 1.00 / (rrf_k + rank)

        by_id = {d["id"]: d for d in docs}
        result = []
        for mid, score in merged.items():
            d = by_id[mid]
            type_bonus = 0.012 if memory_type_hint and d["memory_type"] == memory_type_hint else 0.0
            relation_bonus = 0.008 if relation_hint and d["memory_type"] == "relation" else 0.0
            predicate_bonus = 0.015 if predicate_hint and d["metadata"].get("predicate") == predicate_hint else 0.0
            intent_bonus = 0.010 if intent_hint == "habit" and d["memory_type"] == "rule" else 0.0
            entity_bonus = 0.0
            if relation_hint or memory_type_hint == "relation":
                md = d.get("metadata", {}) or {}
                for key in ("subject", "object", "value"):
                    entity = str(md.get(key) or "").strip()
                    if len(entity) >= 2 and entity in query:
                        entity_bonus = max(entity_bonus, 0.020)
                        break
            active_bonus = 0.005 if d["memory_type"] == "fact" and d["status"] == "active" else 0.0
            conflict_penalty = -0.020 if d["memory_type"] == "fact" and d.get("metadata", {}).get("conflict_status") == "conflict" else 0.0
            deterministic = self._deterministic_signal(query, d)
            temporal_signal = 0.0
            if temporal_start is not None or temporal_end is not None:
                ts = int(d.get("timestamp") or 0)
                in_range = True
                if temporal_start is not None and ts < temporal_start:
                    in_range = False
                if temporal_end is not None and ts > temporal_end:
                    in_range = False
                if in_range:
                    temporal_signal = 0.045
                elif temporal_relation != "after":
                    temporal_signal = -0.018
            structured = score + type_bonus + relation_bonus + predicate_bonus + intent_bonus + entity_bonus + active_bonus + conflict_penalty + deterministic + temporal_signal
            result.append((d, structured))
        result.sort(key=lambda x: x[1], reverse=True)

        if os.getenv("MEMORY_DIAGNOSTICS", "").strip() == "1":
            logger.info(
                "HYBRID_RESULT query=%r top_ids=%s top_types=%s",
                query,
                [d["id"] for d, _ in result[:8]],
                [d["memory_type"] for d, _ in result[:8]],
            )
        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "HYBRID_PROFILE docs=%d bm25=%.2f embedding=%.2f vector_load=%.2f weighted_rrf=%.2f views=message+window+session",
                len(docs), bm25_ms, embedding_ms, vector_load_ms,
                (time.perf_counter() - t_profile) * 1000,
            )
        return result[:max(top_k, 100)]
