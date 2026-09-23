from __future__ import annotations

from collections import defaultdict

from .store import BM25
from .vector import cosine


class HybridRetriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def candidates(self, user_id, query, top_k=30, include_history=False,
                   session_id=None, start_time=None, end_time=None,
                   memory_types=None):
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
                             "object":f["object"],"supersedes_id":f["supersedes_id"]},
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

        bm = BM25()
        bm.fit(docs)
        sparse = bm.search(query, top_k=min(50, len(docs)))
        sparse_rank = {d["id"]: i+1 for i,(d,_) in enumerate(sparse)}

        # Dense retrieval 使用 Add 阶段已经持久化到 SQLite 的向量。
        # 这里只计算一次 query embedding，不再对每条 memory 重新调用 embedding 服务。
        qv = self.embedder.embed(query)
        vectors = self.store.embeddings_by_ids(user_id, [d["id"] for d in docs])
        vector_scores = []
        for d in docs:
            dv = vectors.get(d["id"])
            if dv is None:
                continue
            s = max(-1.0, min(1.0, cosine(qv, dv)))
            vector_scores.append((d, s))
        vector_scores.sort(key=lambda x:x[1], reverse=True)
        dense_rank = {d["id"]: i+1 for i,(d,_) in enumerate(vector_scores[:50])}

        # RRF：避免 sparse/dense 的原始分数不可比。
        rrf_k = 60.0
        merged = defaultdict(float)
        for mid, rank in sparse_rank.items():
            merged[mid] += 1.0 / (rrf_k + rank)
        for mid, rank in dense_rank.items():
            merged[mid] += 1.0 / (rrf_k + rank)

        by_id = {d["id"]: d for d in docs}
        result = []
        for mid, score in merged.items():
            d = by_id[mid]
            # 对当前 active fact 做轻微加权；历史事实除非显式 include_history 不会出现。
            if d["memory_type"] == "fact" and d["status"] == "active":
                score *= 1.05
            result.append((d,score))
        result.sort(key=lambda x:x[1], reverse=True)
        return result[:max(top_k, 30)]
