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
        # SQLite 持久化 + 进程级向量索引：Search 热路径不再反复读取/解析 JSON 向量。
        self.vector_index = MemoryVectorIndex(self.store)
        self.hybrid = HybridRetriever(self.store, self.embedder, self.vector_index)
        self.query_analyzer = QueryAnalyzer()
        self.reranker = LightweightReranker()
        self.evidence = EvidenceBuilder()
        self.evidence_chain = EvidenceChainBuilder(max_hops=3)

    def add(self, request):
        # Claim request_id atomically before doing any writes. This closes the
        # check-then-act race between concurrent duplicate Add requests.
        if not self.store.claim_request(request.request_id, request.user_id):
            return True

        try:
            return self._add_claimed(request)
        except Exception:
            # Do not permanently consume a request_id when the Add operation
            # fails before completion; the caller can retry safely.
            self.store.release_request(request.request_id)
            raise

    def _add_claimed(self, request):
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

                # 原始记忆同时写入 SQLite 和进程级向量索引，保证 Add -> Search 立即可见。
                raw_vector = self.embedder.embed(msg.content)
                self.store.embed(raw_id, request.user_id, raw_vector)
                self.vector_index.add(request.user_id, raw_id, raw_vector)

                role = (msg.role or "user").strip().lower()
                source = "system" if role == "system" else ("assistant" if role in {"assistant", "model"} else "user")

                # Conservative reference resolution before extraction.  We use
                # only recently seen structured entities for this user and do
                # not resolve ambiguous pronouns when no unique anchor exists.
                recent_entities = []
                # Structured relations are the strongest anchors, but a prior
                # message may introduce an entity before any relation exists
                # (e.g. "我最近负责 Acme 项目" -> "这个项目...").
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

        # request_id was already atomically claimed before processing.
        return True

    def search(self, request):
        search_t0 = time.perf_counter()
        query = (request.query or request.question or "").strip()
        if not query:
            return []

        # 用用户已有最新记忆作为相对时间参考，避免服务当前时间与 benchmark 时间轴不一致。
        t0 = time.perf_counter()
        latest = self.store.all_raw(request.user_id)
        latest_ms = (time.perf_counter() - t0) * 1000
        reference_ts = latest[0]["timestamp"] if latest else now_ms()

        t0 = time.perf_counter()
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
        analyze_ms = (time.perf_counter() - t0) * 1000

        # P5: multi-hop retrieval happens BEFORE reranking.  The old P4
        # pipeline reranked first and only then tried to reconstruct a chain,
        # which meant a bridge memory could be discarded before it was usable.
        # We now run bounded iterative retrieval and merge all rounds first.
        t0 = time.perf_counter()
        retrieval_queries = self.query_analyzer.retrieval_queries(plan)
        all_candidates = []
        seen_candidate_ids = set()

        def _merge_candidates(rows):
            for d, score in rows:
                item = dict(d)
                item["score"] = float(score)
                cid = item["id"]
                # Keep the strongest score for the same memory.
                if cid not in seen_candidate_ids:
                    seen_candidate_ids.add(cid)
                    all_candidates.append(item)
                else:
                    for existing in all_candidates:
                        if existing["id"] == cid and score > existing["score"]:
                            existing["score"] = float(score)
                            break

        # First pass: full/decomposed queries. For multi-hop, every planned
        # evidence requirement also receives an independent candidate quota.
        requirement_plans = self.query_analyzer.evidence_requirements(plan) if plan.multi_hop else []
        requirement_by_id = {r["id"]: r for r in requirement_plans}
        # P8-B: apply normalized query time as a retrieval constraint.
        # Open after/after-latest expressions are not used as hard filters.
        temporal_start = request.start_time
        temporal_end = request.end_time
        if temporal_start is None and plan.temporal_relation != "after":
            temporal_start = plan.temporal_start
        if temporal_end is None and plan.temporal_relation != "after":
            temporal_end = plan.temporal_end

        def _merge_candidates(rows, requirement_id=None):
            for d, score in rows:
                item = dict(d)
                item["score"] = float(score)
                item["_evidence_requirements"] = list(item.get("_evidence_requirements", []))
                if requirement_id and requirement_id not in item["_evidence_requirements"]:
                    item["_evidence_requirements"].append(requirement_id)
                cid = item["id"]
                if cid not in seen_candidate_ids:
                    seen_candidate_ids.add(cid)
                    all_candidates.append(item)
                else:
                    for existing in all_candidates:
                        if existing["id"] == cid:
                            existing["_evidence_requirements"] = list(dict.fromkeys(
                                existing.get("_evidence_requirements", []) +
                                item.get("_evidence_requirements", [])
                            ))
                            if score > existing["score"]:
                                existing["score"] = float(score)
                            break

        for rq in retrieval_queries:
            rows = self.hybrid.candidates(
                user_id=request.user_id, query=rq,
                top_k=max(40, request.top_k * 8),
                include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
                session_id=request.session_id, start_time=temporal_start, end_time=temporal_end,
                memory_types=request.memory_types, memory_type_hint=plan.memory_type_hint,
                temporal_relation=plan.temporal_relation, relation_hint=plan.relation_hint,
                sparse_query=rq, predicate_hint=plan.predicate_hint, intent_hint=plan.intent_hint,
            )
            _merge_candidates(rows)

        if requirement_plans:
            per_requirement_k = max(24, min(60, request.top_k * 2))
            for req in requirement_plans:
                rq = req["query"]
                rows = self.hybrid.candidates(
                    user_id=request.user_id, query=rq, top_k=per_requirement_k,
                    include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
                    session_id=request.session_id, start_time=request.start_time, end_time=request.end_time,
                    memory_types=request.memory_types, memory_type_hint=plan.memory_type_hint,
                    temporal_relation=plan.temporal_relation, relation_hint=plan.relation_hint,
                    sparse_query=rq, predicate_hint=plan.predicate_hint, intent_hint=plan.intent_hint,
                )
                _merge_candidates(rows, req["id"])

        candidates = [(x, x["score"]) for x in all_candidates]
        hybrid_ms = (time.perf_counter() - t0) * 1000

        second_round_ms = 0.0

        if plan.multi_hop and all_candidates:
            # P5 iterative retrieval: each round extracts entities from the
            # current evidence and uses them as the next-hop query.  This works
            # across sessions because all memories are scoped by user_id, not
            # by session_id unless the caller explicitly requests a session.
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
                # Add relation semantics so an entity-only second hop still
                # retrieves workplace/headquarters/causal evidence.
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

                # Expand from the newly found structured entities.
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

        candidates = [(x, x["score"]) for x in all_candidates]
        result = list(all_candidates)
        # 查询级时间约束：优先使用显式时间窗口；“以前/去年/上个月”等
        # 会由 QueryAnalyzer 归一化后应用到候选证据。
        if plan.temporal and (plan.temporal_start is not None or plan.temporal_end is not None):
            result = [
                r for r in result
                if (plan.temporal_start is None or r.get("valid_from", r.get("timestamp", 0)) >= plan.temporal_start)
                and (plan.temporal_end is None or r.get("valid_from", r.get("timestamp", 0)) <= plan.temporal_end)
            ]

        # 去重
        dedup = {}
        for r in result:
            key = (r["content"].strip(), r.get("memory_type"))
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        ranked = list(dedup.values())

        # P8 causal evidence assembly: once a causal query has a seed
        # candidate, walk the stored causes edges for up to three hops and
        # inject the missing bridge/outcome evidence before reranking.  This is
        # deliberately evidence retrieval, not answer generation.
        causal_query = any(
            marker in query
            for marker in ("为什么", "为何", "原因", "导致", "因为", "所以", "因此",
                           "how did", "why", "cause", "caused", "because")
        )
        if causal_query and ranked:
            causal_req_ids = [
                req["id"] for req in requirement_plans
                if str(req.get("kind", "")).startswith("causal_")
            ]
            relation_rows = [
                rel for rel in self.store.relations(request.user_id)
                if rel.get("predicate") == "causes"
            ]
            frontier = set()
            for item in ranked[:40]:
                md = item.get("metadata", {}) or {}
                for key in ("subject", "object", "value", "event"):
                    value = str(md.get(key) or "").strip()
                    if value and value not in {"user", "我", "用户"}:
                        frontier.add(value)

            # Also seed from causal relation endpoints already retrieved.
            for rel in relation_rows:
                if rel.get("subject") in frontier or rel.get("object") in frontier:
                    frontier.add(str(rel.get("subject") or "").strip())
                    frontier.add(str(rel.get("object") or "").strip())

            selected_relations = []
            seen_rel_ids = set()
            for _hop in range(3):
                if not frontier:
                    break
                next_frontier = set()
                for rel in relation_rows:
                    rid = rel.get("id")
                    subject = str(rel.get("subject") or "").strip()
                    object_ = str(rel.get("object") or "").strip()
                    if rid in seen_rel_ids or not subject or not object_:
                        continue
                    if subject in frontier or object_ in frontier:
                        selected_relations.append(rel)
                        seen_rel_ids.add(rid)
                        next_frontier.update({subject, object_})
                if not next_frontier:
                    break
                frontier.update(next_frontier)

            existing_ids = {item.get("id") for item in ranked}
            raw_by_content = {
                str(row.get("content") or "").strip(): row
                for row in self.store.all_raw(request.user_id)
            }
            for rel in selected_relations[:16]:
                rid = rel.get("id")
                if rid in existing_ids:
                    continue
                item = {
                    "id": rid,
                    "content": rel.get("content", ""),
                    "role": "relation",
                    "timestamp": rel.get("timestamp", 0),
                    "user_id": request.user_id,
                    "session_id": "",
                    "score": 0.42,
                    "source": "causal_chain",
                    "memory_type": "relation",
                    "status": "active",
                    "valid_from": rel.get("timestamp", 0),
                    "valid_to": None,
                    "metadata": {
                        "subject": rel.get("subject"),
                        "predicate": "causes",
                        "object": rel.get("object"),
                        "causal_hop_evidence": True,
                        "_evidence_requirements": list(causal_req_ids),
                    },
                }
                ranked.append(item)
                existing_ids.add(rid)

                raw = raw_by_content.get(str(rel.get("content") or "").strip())
                if raw and raw["id"] not in existing_ids:
                    ranked.append({
                        "id": raw["id"],
                        "content": raw["content"],
                        "role": raw["role"],
                        "timestamp": raw["timestamp"],
                        "user_id": request.user_id,
                        "session_id": raw["session_id"],
                        "score": 0.40,
                        "source": "causal_provenance",
                        "memory_type": "raw",
                        "status": "active",
                        "valid_from": raw["timestamp"],
                        "valid_to": None,
                        "metadata": {
                            "request_id": raw["request_id"],
                            "source_message_ids": [raw["id"]],
                            "causal_parent_id": rid,
                            "_evidence_requirements": list(causal_req_ids),
                        },
                    })
                    existing_ids.add(raw["id"])

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
        # Coverage-aware reranking: candidates retrieved for a specific
        # requirement must also be judged against that requirement. Otherwise
        # a candidate can be retrieved correctly (e.g. "John") and then lose
        # to generic candidates when the whole question is reranked.
        if requirement_by_id and ranked:
            for item in ranked:
                req_ids = item.get("_evidence_requirements", []) or []
                if not req_ids:
                    continue
                best_req_score = 0.0
                for req_id in req_ids:
                    req = requirement_by_id.get(req_id)
                    if not req:
                        continue
                    req_score = self.reranker.score(req["query"], item.get("content", ""))
                    best_req_score = max(best_req_score, req_score)
                item["_requirement_score"] = round(best_req_score, 6)
                item["score"] = round(
                    0.70 * float(item.get("score", 0.0))
                    + 0.30 * best_req_score,
                    6,
                )

            # Seed one best candidate for each requirement. This ordering is
            # consumed by EvidenceBuilder, which now explicitly protects the
            # requirement coverage slots before global fill.
            ordered = []
            used = set()
            for req_id in requirement_by_id:
                best = max(
                    (
                        item for item in ranked
                        if item.get("id") not in used
                        and req_id in (item.get("_evidence_requirements", []) or [])
                    ),
                    key=lambda item: (
                        float(item.get("_requirement_score", 0.0)),
                        float(item.get("score", 0.0)),
                    ),
                    default=None,
                )
                if best is not None:
                    ordered.append(best)
                    used.add(best.get("id"))
            ordered.extend(item for item in ranked if item.get("id") not in used)
            ranked = ordered

            if os.getenv("MEMORY_DIAGNOSTICS", "").strip() == "1":
                logger.info(
                    "REQUIREMENT_DIAGNOSTICS %s",
                    [
                        {
                            "id": req_id,
                            "query": requirement_by_id[req_id]["query"],
                            "candidates": sum(
                                1 for item in ranked
                                if req_id in (item.get("_evidence_requirements", []) or [])
                            ),
                            "best_score": max(
                                [
                                    float(item.get("_requirement_score", 0.0))
                                    for item in ranked
                                    if req_id in (item.get("_evidence_requirements", []) or [])
                                ] or [0.0]
                            ),
                        }
                        for req_id in requirement_by_id
                    ],
                )
        # Evidence completeness: whenever a structured memory survives
        # reranking, explicitly carry its source raw message into the final
        # candidate pool. The benchmark evaluates message-level evidence, so
        # a fact/relation hit must not hide the original supporting turn.
        if ranked:
            source_ids = []
            for item in ranked[:max(40, min(request.top_k, 60))]:
                for source_id in (item.get("metadata", {}) or {}).get("source_message_ids", []):
                    if source_id not in source_ids:
                        source_ids.append(source_id)
            if source_ids:
                raw_rows = {
                    row["id"]: row
                    for row in self.store.raw_by_ids(request.user_id, source_ids)
                }
                existing_ids = {item.get("id") for item in ranked}
                evidence_expansions = []
                for parent in ranked[:max(40, min(request.top_k, 60))]:
                    parent_score = float(parent.get("score", 0.0))
                    parent_sources = (parent.get("metadata", {}) or {}).get("source_message_ids", [])
                    for source_id in parent_sources[:2]:
                        raw = raw_rows.get(source_id)
                        if not raw or source_id in existing_ids:
                            continue
                        evidence_expansions.append({
                            "id": raw["id"],
                            "content": raw["content"],
                            "role": raw["role"],
                            "timestamp": raw["timestamp"],
                            "user_id": request.user_id,
                            "session_id": raw["session_id"],
                            "score": parent_score + 0.003,
                            "source": "evidence_provenance",
                            "memory_type": "raw",
                            "status": "active",
                            "valid_from": raw["timestamp"],
                            "valid_to": None,
                            "metadata": {
                                "request_id": raw["request_id"],
                                "source_message_ids": [raw["id"]],
                                "view": "message",
                                "evidence_parent_id": parent.get("id"),
                            },
                        })
                        existing_ids.add(source_id)
                if evidence_expansions:
                    ranked.extend(evidence_expansions)
                    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        # P5: preserve one deterministic next-hop bridge after reranking.
        if plan.multi_hop and ranked:
            relation_rows = self.store.relations(request.user_id)
            ranked_entities = set()
            for item in ranked:
                md = item.get("metadata", {}) or {}
                for key in ("subject", "object", "value", "event"):
                    value = str(md.get(key) or "").strip()
                    if value and value not in {"user", "我", "用户"}:
                        ranked_entities.add(value)
            if ranked_entities:
                relevant_predicates = {"belongs_to", "causes", "headquarters", "work_at", "recommend", "located_in", "alias_of"}
                existing_ids = {item.get("id") for item in ranked}

                # Choose the bridge according to the query's requested final
                # relation, not SQLite insertion order.  For example,
                # “总部所在城市属于哪个国家” must keep the belongs_to edge
                # “上海属于中国”, even when distractor relations were added
                # earlier or later.
                if any(x in query for x in ("属于哪个国家", "属于什么国家", "哪个国家")):
                    predicate_priority = {"belongs_to": 0, "located_in": 1, "headquarters": 2}
                elif any(x in query for x in ("为什么", "原因", "导致", "因为", "所以", "因此")):
                    predicate_priority = {"causes": 0}
                elif any(x in query for x in ("总部", "公司")):
                    predicate_priority = {"headquarters": 0, "work_at": 1, "located_in": 2}
                elif any(x in query for x in ("推荐", "介绍")):
                    predicate_priority = {"recommend": 0}
                else:
                    predicate_priority = {}

                bridge_candidates = []
                for rel in relation_rows:
                    predicate = rel.get("predicate")
                    if predicate not in relevant_predicates:
                        continue
                    subject = str(rel.get("subject") or "").strip()
                    object_ = str(rel.get("object") or "").strip()
                    if not ({subject, object_} & ranked_entities):
                        continue

                    # Do not exclude an already-ranked relation here.  It may
                    # have been present in the reranker output but later lost
                    # from the EvidenceBuilder prefix.  We explicitly promote
                    # the semantic final-hop relation back to the front.
                    bridge_candidates.append((predicate_priority.get(predicate, 50), rel))

                bridge = None
                if bridge_candidates:
                    _, rel = min(
                        bridge_candidates,
                        key=lambda pair: (pair[0], str(pair[1].get("id") or "")),
                    )
                    subject = str(rel.get("subject") or "").strip()
                    object_ = str(rel.get("object") or "").strip()
                    bridge = {
                        "id": rel["id"], "content": rel["content"], "role": "relation",
                        "timestamp": rel["timestamp"], "user_id": request.user_id,
                        "session_id": "", "score": 0.40, "source": "next_hop_bridge",
                        "memory_type": "relation", "status": "active",
                        "valid_from": rel["timestamp"], "valid_to": None,
                        "metadata": {"subject": subject, "predicate": rel.get("predicate"), "object": object_, "graph_hop": 1},
                    }

                if bridge is not None:
                    # EvidenceBuilder keeps the first top_k candidates, so a
                    # bridge appended at the tail would still be discarded.
                    bridge["score"] = max(bridge["score"], 0.90)
                    ranked = [bridge] + [item for item in ranked if item.get("id") != bridge["id"]]


        rerank_ms = (time.perf_counter() - t0) * 1000

        # P2：当前状态优先。对于“现在/目前/当前/最新”等状态查询，
        # 先按 predicate 命中的 active fact/profile 排在原始历史消息之前。
        # 这避免旧 raw memory 因词面相似度压过已经被治理为 current 的新值。
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

        # P4：多跳查询按“证据链”而不是单条 memory 独立排序。
        # 仅在多跳查询启用；保持 P2 的默认查询路径不变。
        chain_ms = 0.0
        if plan.multi_hop:
            t0 = time.perf_counter()
            ranked = self.evidence_chain.annotate(plan.rewritten, ranked)
            chain_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked = self.evidence.build(ranked, request.top_k)
        evidence_ms = (time.perf_counter() - t0) * 1000

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

        total_ms = (time.perf_counter() - search_t0) * 1000
        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "SEARCH_PROFILE query=%r total=%.2f latest=%.2f analyze=%.2f hybrid=%.2f "
                "second_round=%.2f rerank=%.2f chain=%.2f evidence=%.2f candidates=%d final=%d",
                query, total_ms, latest_ms, analyze_ms, hybrid_ms, second_round_ms,
                rerank_ms, chain_ms, evidence_ms, len(candidates), len(ranked),
            )
        return ranked[:request.top_k]
