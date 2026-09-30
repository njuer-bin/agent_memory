from __future__ import annotations

import logging
import os
import time
from collections import defaultdict


logger = logging.getLogger(__name__)

from .store import BM25
from .vector import cosine
from .vector_index import MemoryVectorIndex


class HybridRetriever:
    def __init__(self, store, embedder, vector_index: MemoryVectorIndex | None = None):
        self.store = store
        self.embedder = embedder
        self.vector_index = vector_index or MemoryVectorIndex(store)

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

        docs = []
        for r in raws:
            docs.append({
                "id": r["id"], "content": r["content"], "role": r["role"],
                "timestamp": r["timestamp"], "user_id": user_id,
                "session_id": r["session_id"], "memory_type": "raw",
                "status": "active", "source": "raw",
                "valid_from": r["timestamp"], "valid_to": None,
                "metadata": {"request_id": r["request_id"]},
            })

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
                "timestamp": r["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "relation",
                "status": "active", "source": "graph",
                "valid_from": r["timestamp"], "valid_to": None,
                "metadata": {"subject":r["subject"],"predicate":r["predicate"],
                             "object":r["object"]},
            })

        for r in rules:
            docs.append({
                "id": r["id"], "content": r["content"], "role": "rule",
                "timestamp": r["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "rule",
                "status": "active", "source": "rule",
                "valid_from": r["timestamp"], "valid_to": None,
                "metadata": {},
            })

        for p in profiles:
            docs.append({
                "id": f"profile:{p['user_id']}:{p['key']}",
                "content": p["content"], "role": "profile",
                "timestamp": p["timestamp"], "user_id": user_id,
                "session_id": "", "memory_type": "profile",
                "status": "active", "source": "profile",
                "valid_from": p["timestamp"], "valid_to": None,
                "metadata": {"key":p["key"],"value":p["value"]},
            })

        if memory_types:
            allowed = set(memory_types)
            docs = [d for d in docs if d["memory_type"] in allowed]

        if not docs:
            return []

        t_profile = time.perf_counter()
        bm = BM25()
        bm.fit(docs)
        sparse = bm.search(sparse_query or query, top_k=min(50, len(docs)))
        bm25_ms = (time.perf_counter() - t_profile) * 1000
        sparse_rank = {d["id"]: i+1 for i,(d,_) in enumerate(sparse)}

        # Dense retrieval:
        # 第一轮默认 dense+BM25；第二轮可选择 BM25-only，避免再次调用 embedding。
        embedding_ms = 0.0
        vector_load_ms = 0.0
        dense_rank = {}
        if use_dense:
            t_profile = time.perf_counter()
            qv = self.embedder.embed(query)
            embedding_ms = (time.perf_counter() - t_profile) * 1000

            t_profile = time.perf_counter()
            dense_pairs = self.vector_index.search(
                user_id,
                qv,
                ids=[d["id"] for d in docs],
                top_k=min(50, len(docs)),
            )
            vector_load_ms = (time.perf_counter() - t_profile) * 1000
            dense_rank = {mid: i + 1 for i, (mid, _) in enumerate(dense_pairs)}

        # RRF：避免 sparse/dense 的原始分数不可比。
        # P3 实验默认关闭；只有显式设置 RRF_MODE 才启用 Weighted RRF，
        # 这样线上 P1 基线不会被实验代码改变。
        t_profile = time.perf_counter()
        rrf_k = 60.0
        rrf_mode = os.getenv("RRF_MODE", "standard").strip().lower()
        bm25_weight = 1.0
        dense_weight = 1.0
        if rrf_mode == "weighted_v1":
            # 事实型查询更依赖明确词面/实体，语义型查询更依赖 dense。
            if intent_hint == "fact":
                bm25_weight, dense_weight = 0.65, 0.35
            else:
                bm25_weight, dense_weight = 0.35, 0.65
        elif rrf_mode == "weighted_v2":
            # 更激进的实验版本，仅用于 benchmark，不作为默认线上配置。
            if intent_hint == "fact":
                bm25_weight, dense_weight = 0.75, 0.25
            else:
                bm25_weight, dense_weight = 0.25, 0.75

        merged = defaultdict(float)
        for mid, rank in sparse_rank.items():
            merged[mid] += bm25_weight / (rrf_k + rank)
        for mid, rank in dense_rank.items():
            merged[mid] += dense_weight / (rrf_k + rank)

        by_id = {d["id"]: d for d in docs}
        result = []
        for mid, score in merged.items():
            d = by_id[mid]
            type_bonus = 0.0
            if memory_type_hint and d["memory_type"] == memory_type_hint:
                type_bonus = 0.012
            relation_bonus = 0.0
            if relation_hint and d["memory_type"] == "relation":
                relation_bonus = 0.008
            predicate_bonus = 0.0
            if predicate_hint and d["metadata"].get("predicate") == predicate_hint:
                predicate_bonus = 0.015
            intent_bonus = 0.0
            if intent_hint == "habit" and d["memory_type"] == "rule":
                intent_bonus = 0.010
            active_bonus = 0.0
            if d["memory_type"] == "fact" and d["status"] == "active":
                active_bonus = 0.005
            conflict_penalty = 0.0
            if d["memory_type"] == "fact" and d.get("metadata", {}).get("conflict_status") == "conflict":
                conflict_penalty = -0.020
            structured = score + type_bonus + relation_bonus + predicate_bonus + intent_bonus + active_bonus + conflict_penalty
            result.append((d, structured))
        result.sort(key=lambda x:x[1], reverse=True)
        rrf_ms = (time.perf_counter() - t_profile) * 1000
        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "HYBRID_PROFILE docs=%d bm25=%.2f embedding=%.2f vector_load=%.2f rrf=%.2f",
                len(docs), bm25_ms, embedding_ms, vector_load_ms, rrf_ms,
            )
        return result[:max(top_k, 30)]
