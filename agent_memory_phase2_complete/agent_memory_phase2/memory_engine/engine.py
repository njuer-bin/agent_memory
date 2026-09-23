from __future__ import annotations

import time
from typing import Any

from .analyzer import MemoryAnalyzer
from .evidence import EvidenceBuilder
from .governance import MemoryGovernance
from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridRetriever
from .models import new_id, now_ms
from .query_analyzer import QueryAnalyzer
from .reranker import LightweightReranker
from .store import SQLiteStore
from .vector import EmbeddingProvider


class MemoryEngine:
    def __init__(self, db_path="data/memory.db"):
        self.store = SQLiteStore(db_path)
        self.analyzer = MemoryAnalyzer()
        self.governance = MemoryGovernance(self.store)
        self.embedder = EmbeddingProvider()
        self.hybrid = HybridRetriever(self.store, self.embedder)
        self.graph = GraphRetriever(self.store)
        self.query_analyzer = QueryAnalyzer()
        self.reranker = LightweightReranker()
        self.evidence = EvidenceBuilder()

    def add(self, request):
        if self.store.request_seen(request.request_id):
            return True

        # 按 20 条消息或约 2000 词做确定性批次边界。
        batches = []
        current = []
        words = 0
        for msg in request.messages:
            n = len(msg.content.split())
            if current and (len(current) >= 20 or words + n > 2000):
                batches.append(current)
                current, words = [], 0
            current.append(msg)
            words += n
        if current:
            batches.append(current)

        for batch_idx, batch in enumerate(batches):
            for msg_idx, msg in enumerate(batch):
                ts = msg.timestamp or now_ms()
                raw_id = new_id("raw")
                self.store.insert_raw({
                    "id": raw_id,
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": ts,
                    "chunk_index": batch_idx,
                })

                # 原始记忆也建立向量索引，确保无规则抽取时仍可召回。
                self.store.embed(raw_id, request.user_id, self.embedder.embed(msg.content))

                analyzed = self.analyzer.analyze(
                    request.user_id, msg.content, ts
                )

                for fact in analyzed["facts"]:
                    inserted, _ = self.governance.accept_fact(fact)
                    if inserted:
                        self.store.embed(fact.id, request.user_id,
                                         self.embedder.embed(fact.content))

                for rel in analyzed["relations"]:
                    self.store.insert_relation(rel)
                    self.store.embed(rel.id, request.user_id,
                                     self.embedder.embed(rel.content))

                for event in analyzed["events"]:
                    self.store.insert_event(event)
                    self.store.embed(event.id, request.user_id,
                                     self.embedder.embed(event.content))

                for rule in analyzed["rules"]:
                    self.store.insert_rule(rule)
                    self.store.embed(rule.id, request.user_id,
                                     self.embedder.embed(rule.content))

                for profile in analyzed["profiles"]:
                    self.store.upsert_profile(profile)

        # 只有所有数据都成功写入并建立索引后才登记 request_id。
        self.store.register_request(request.request_id, request.user_id)
        return True

    def search(self, request):
        query = (request.query or request.question or "").strip()
        if not query:
            return []

        # 用用户已有最新记忆作为相对时间参考，避免服务当前时间与 benchmark 时间轴不一致。
        latest = self.store.all_raw(request.user_id)
        reference_ts = latest[0]["timestamp"] if latest else now_ms()
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
        t0 = time.perf_counter()

        candidates = self.hybrid.candidates(
            user_id=request.user_id,
            query=plan.rewritten,
            top_k=max(30, request.top_k * 5),
            include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
            session_id=request.session_id,
            start_time=request.start_time,
            end_time=request.end_time,
            memory_types=request.memory_types,
        )

        result = []
        for d, score in candidates:
            item = dict(d)
            item["score"] = score
            result.append(item)

        # 查询级时间约束：优先使用显式时间窗口；“以前/去年/上个月”等
        # 会由 QueryAnalyzer 归一化后应用到候选证据。
        if plan.temporal and (plan.temporal_start is not None or plan.temporal_end is not None):
            result = [
                r for r in result
                if (plan.temporal_start is None or r.get("valid_from", r.get("timestamp", 0)) >= plan.temporal_start)
                and (plan.temporal_end is None or r.get("valid_from", r.get("timestamp", 0)) <= plan.temporal_end)
            ]

        # 仅多跳查询执行额外图扩展，避免所有查询都增加延迟。
        if plan.multi_hop:
            expanded = self.graph.expand(request.user_id, result[:5], limit=10)
            result.extend(expanded)

        # 去重
        dedup = {}
        for r in result:
            key = (r["content"].strip(), r.get("memory_type"))
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        ranked = list(dedup.values())
        ranked = self.reranker.rerank(plan.rewritten, ranked, max(request.top_k * 3, request.top_k))
        ranked = self.evidence.build(ranked, request.top_k)

        # 时间查询的结果顺序：当前有效事实优先；历史查询保留时间信息。
        if not request.include_history and plan.temporal:
            ranked.sort(
                key=lambda x: (
                    1 if x.get("status") == "active" else 0,
                    x.get("timestamp", 0),
                    x.get("score", 0.0),
                ),
                reverse=True,
            )

        return ranked[:request.top_k]
