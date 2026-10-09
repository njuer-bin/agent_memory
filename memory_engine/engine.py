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
        # SQLite 持久化 + 进程级向量索引：Search 热路径不再反复读取/解析 JSON 向量。
        self.vector_index = MemoryVectorIndex(self.store)
        self.hybrid = HybridRetriever(self.store, self.embedder, self.vector_index)
        self.query_analyzer = QueryAnalyzer()
        self.reranker = LightweightReranker()
        self.evidence = EvidenceBuilder()
        self.evidence_chain = EvidenceChainBuilder(max_hops=3)
        self.memory_links = MemoryLinkBuilder()

    def add(self, request):
        # AML Add uses the LLM extraction path by default. The local fallback
        # can still be enabled explicitly for offline development/tests with
        # AML_USE_LLM=0. llm_add.py owns the request claim/release lifecycle.
        use_llm = os.getenv("AML_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
        if use_llm:
            from .llm_add import add_with_llm
            return add_with_llm(self, request)

        # Deterministic fallback: claim request_id atomically before writes.
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

                # P8-A: persist only high-confidence provenance links. Raw
                # messages remain canonical; links are an auxiliary path layer.
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
