from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from .analyzer import MemoryAnalyzer
from .evidence import EvidenceBuilder
from .evidence_chain import EvidenceChainBuilder
from .governance import MemoryGovernance
from .hybrid_retriever import HybridRetriever
from .memory_links import MemoryLinkBuilder
from .models import new_id, now_ms
from .query_analyzer import QueryAnalyzer
from .reranker import LightweightReranker
from .store import SQLiteStore
from .vector import EmbeddingProvider
from .vector_index import MemoryVectorIndex


logger = logging.getLogger(__name__)


class MemoryEngine:
    def __init__(self, db_path="data/memory.db"):
        self.store = SQLiteStore(db_path)
        self.analyzer = MemoryAnalyzer()
        self.governance = MemoryGovernance(self.store)
        self.embedder = EmbeddingProvider()
        self.vector_index = MemoryVectorIndex(self.store)
        self.hybrid = HybridRetriever(self.store, self.embedder, self.vector_index)
        self.query_analyzer = QueryAnalyzer()
        self.reranker = LightweightReranker()
        self.evidence = EvidenceBuilder()
        self.evidence_chain = EvidenceChainBuilder(max_hops=3)
        self.memory_links = MemoryLinkBuilder()

    def add(self, request):
        use_llm = os.getenv("AML_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
        if use_llm:
            from .llm_add import add_with_llm
            return add_with_llm(self, request)

        if not self.store.claim_request(request.request_id, request.user_id):
            return True

        try:
            return self._add_claimed(request)
        except Exception:
            self.store.release_request(request.request_id)
            raise

    def _add_claimed(self, request):
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

                previous_raw = [
                    row for row in self.store.all_raw(request.user_id, request.session_id)
                    if row.get("id") != raw_id
                ][:8]
                for link in self.memory_links.build(
                    {"id": raw_id, "user_id": request.user_id, "session_id": request.session_id,
                     "content": msg.content, "timestamp": ts},
                    previous_raw,
                ):
                    link["id"] = new_id("link")
                    link["user_id"] = request.user_id
                    self.store.insert_memory_link(link)

                raw_vector = self.embedder.embed(msg.content)
                self.store.embed(raw_id, request.user_id, raw_vector)
                self.vector_index.add(request.user_id, raw_id, raw_vector)

                role = (msg.role or "user").strip().lower()
                source = "system" if role == "system" else ("assistant" if role in {"assistant", "model"} else "user")

                recent_entities = []
                for raw in self.store.all_raw(request.user_id)[:12]:
                    raw_content = str(raw.get("content") or "")
                    for match in re.findall(r"([A-Za-z][A-Za-z0-9_-]{1,39})\s*(?=项目|公司|集团|团队)", raw_content):
                        if match not in recent_entities:
                            recent_entities.append(match)
                for rel in self.store.relations(request.user_id)[:12]:
                    for value in (rel.get("subject"), rel.get("object")):
                        value = str(value or "").strip()
                        if value and value not in {"user", "我", "用户"} and value not in recent_entities:
                            recent_entities.append(value)
                resolved_content = self.analyzer.resolve_references(msg.content, recent_entities[:1])
                analyzed = self.analyzer.analyze(
                    request.user_id, resolved_content, ts, source=source
                )

                for fact in analyzed["facts"]:
                    inserted, _ = self.governance.accept_fact(fact)
                    if inserted:
                        fact_vector = self.embedder.embed(fact.content)
                        self.store.embed(fact.id, request.user_id, fact_vector)
                        self.vector_index.add(request.user_id, fact.id, fact_vector)

                for rel in analyzed["relations"]:
                    self.store.insert_relation(rel)
                    rel_vector = self.embedder.embed(rel.content)
                    self.store.embed(rel.id, request.user_id, rel_vector)
                    self.vector_index.add(request.user_id, rel.id, rel_vector)

                for event in analyzed["events"]:
                    self.store.insert_event(event)
                    event_vector = self.embedder.embed(event.content)
                    self.store.embed(event.id, request.user_id, event_vector)
                    self.vector_index.add(request.user_id, event.id, event_vector)

                for rule in analyzed["rules"]:
                    self.store.insert_rule(rule)
                    rule_vector = self.embedder.embed(rule.content)
                    self.store.embed(rule.id, request.user_id, rule_vector)
                    self.vector_index.add(request.user_id, rule.id, rule_vector)

                for profile in analyzed["profiles"]:
                    self.store.upsert_profile(profile)

        return True

    def search(self, request):
        search_t0 = time.perf_counter()
        query = (request.query or request.question or "").strip()
        if not query:
            return []

        t0 = time.perf_counter()
        latest = self.store.all_raw(request.user_id)
        latest_ms = (time.perf_counter() - t0) * 1000
        reference_ts = latest[0]["timestamp"] if latest else now_ms()

        t0 = time.perf_counter()
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
        analyze_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retrieval_queries = self.query_analyzer.retrieval_queries(plan)
        all_candidates = []
        seen_candidate_ids = set()

        def _merge_candidates(rows):
            for d, score in rows:
                item = dict(d)
                item["score"] = float(score)
                cid = item["id"]
                if cid not in seen_candidate_ids:
                    seen_candidate_ids.add(cid)
                    all_candidates.append(item)
                else:
                    for existing in all_candidates:
                        if existing["id"] == cid and score > existing["score"]:
                            existing["score"] = float(score)
                            break

        for rq in retrieval_queries:
            rows = self.hybrid.candidates(
                user_id=request.user_id,
                query=rq,
                top_k=max(40, request.top_k * 8),
                include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
                session_id=request.session_id,
                start_time=request.start_time,
                end_time=request.end_time,
                memory_types=request.memory_types,
                memory_type_hint=plan.memory_type_hint,
                temporal_relation=plan.temporal_relation,
                relation_hint=plan.relation_hint,
                sparse_query=rq,
                predicate_hint=plan.predicate_hint,
                intent_hint=plan.intent_hint,
            )
            _merge_candidates(rows)

        candidates = [(x, x["score"]) for x in all_candidates]
        hybrid_ms = (time.perf_counter() - t0) * 1000

        second_round_ms = 0.0

        if plan.multi_hop and all_candidates:
            relation_rows = self.store.relations(request.user_id)
            seed_entities = set()
            for item in all_candidates[:40]:
                md = item.get("metadata", {}) or {}
                for key in ("subject", "object", "value", "event"):
                    value = str(md.get(key) or "").strip()
                    if value and value not in {"user", "我", "用户"}:
                        seed_entities.add(value)
            for rel in relation_rows:
                subject = str(rel.get("subject") or "").strip()
                object_ = str(rel.get("object") or "").strip()
                if not ({subject, object_} & seed_entities):
                    continue
                item = {
                    "id": rel["id"], "content": rel["content"], "role": "relation",
                    "timestamp": rel["timestamp"], "user_id": request.user_id,
                    "session_id": "", "score": 0.030, "source": "structured_bridge",
                    "memory_type": "relation", "status": "active",
                    "valid_from": rel["timestamp"], "valid_to": None,
                    "metadata": {"subject": subject, "predicate": rel.get("predicate"),
                                 "object": object_, "graph_hop": 1},
                }
                _merge_candidates([(item, item["score"])])

            frontier_entities = []
            for item in all_candidates[:12]:
                md = item.get("metadata", {}) or {}
                for key in ("subject", "object", "value", "event"):
                    value = str(md.get(key) or "").strip()
                    if value and value not in {"user", "我", "用户"} and value not in frontier_entities:
                        frontier_entities.append(value)

            for hop in range(1, 4):
                if not frontier_entities:
                    break
                round_query = " ".join(frontier_entities[:8])
                if any(x in query for x in ("总部", "公司", "工作")):
                    round_query += " 工作 公司 总部 位于"
                elif any(x in query for x in ("推荐", "介绍")):
                    round_query += " 推荐 介绍"
                elif any(x in query for x in ("为什么", "原因", "导致", "因为", "所以", "因此")):
                    round_query += " 原因 导致 因果 结果"

                t_round = time.perf_counter()
                rows = self.hybrid.candidates(
                    user_id=request.user_id,
                    query=round_query,
                    top_k=max(30, request.top_k * 6),
                    include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
                    session_id=request.session_id,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    memory_types=request.memory_types,
                    memory_type_hint="relation",
                    temporal_relation=plan.temporal_relation,
                    relation_hint=True,
                    sparse_query=round_query,
                    predicate_hint=None,
                    intent_hint="relation",
                    use_dense=False,
                )
                _merge_candidates(rows)
                second_round_ms += (time.perf_counter() - t_round) * 1000

                next_entities = []
                for item, _score in rows[:15]:
                    md = item.get("metadata", {}) or {}
                    for key in ("subject", "object", "value"):
                        value = str(md.get(key) or "").strip()
                        if value and value not in {"user", "我", "用户"} and value not in frontier_entities and value not in next_entities:
                            next_entities.append(value)
                if not next_entities:
                    break
                frontier_entities.extend(next_entities[:8])

        result = list(all_candidates)
        if plan.temporal and (plan.temporal_start is not None or plan.temporal_end is not None):
            result = [
                r for r in result
                if (plan.temporal_start is None or r.get("valid_from", r.get("timestamp", 0)) >= plan.temporal_start)
                and (plan.temporal_end is None or r.get("valid_from", r.get("timestamp", 0)) <= plan.temporal_end)
            ]

        dedup = {}
        for r in result:
            key = (r["content"].strip(), r.get("memory_type"))
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        ranked = list(dedup.values())
        t0 = time.perf_counter()
        ranked = self.reranker.rerank(
            plan.rewritten,
            ranked,
            max(request.top_k * 3, request.top_k),
            memory_type_hint=plan.memory_type_hint,
            relation_hint=plan.relation_hint,
            temporal_relation=plan.temporal_relation,
            expanded_query=plan.expanded_query or plan.rewritten,
            predicate_hint=plan.predicate_hint,
            intent_hint=plan.intent_hint,
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        current_markers = ("现在", "目前", "当前", "最新", "现居", "如今")
        if (
            not request.include_history
            and plan.predicate_hint
            and any(marker in query for marker in current_markers)
        ):
            def _current_state_key(item):
                md = item.get("metadata", {}) or {}
                is_matching_fact = (
                    item.get("memory_type") == "fact"
                    and item.get("status") == "active"
                    and md.get("predicate") == plan.predicate_hint
                )
                is_matching_profile = (
                    item.get("memory_type") == "profile"
                    and md.get("key") == plan.predicate_hint
                )
                return (
                    1 if (is_matching_fact or is_matching_profile) else 0,
                    1 if item.get("memory_type") == "fact" and item.get("status") == "active" else 0,
                    item.get("timestamp", 0),
                    item.get("score", 0.0),
                )
            ranked.sort(key=_current_state_key, reverse=True)

        chain_ms = 0.0
        if plan.multi_hop:
            t0 = time.perf_counter()
            ranked = self.evidence_chain.annotate(plan.rewritten, ranked)
            chain_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked = self.evidence.build(ranked, request.top_k)
        evidence_ms = (time.perf_counter() - t0) * 1000

        if not request.include_history and plan.temporal:
            ranked.sort(
                key=lambda x: (
                    1 if x.get("status") == "active" else 0,
                    x.get("timestamp", 0),
                    x.get("score", 0.0),
                ),
                reverse=True,
            )

        total_ms = (time.perf_counter() - search_t0) * 1000
        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "SEARCH_PROFILE query=%r total=%.2f latest=%.2f analyze=%.2f hybrid=%.2f "
                "second_round=%.2f rerank=%.2f chain=%.2f evidence=%.2f candidates=%d final=%d",
                query, total_ms, latest_ms, analyze_ms, hybrid_ms, second_round_ms,
                rerank_ms, chain_ms, evidence_ms, len(candidates), len(ranked),
            )
        return ranked[:request.top_k]
