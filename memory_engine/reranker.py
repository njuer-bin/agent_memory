from __future__ import annotations

import logging
import os
import time

from .store import tokenize


logger = logging.getLogger(__name__)


class LightweightReranker:
    """
    无需下载模型的可解释 fallback。
    如果以后接入 BGE reranker，只需替换 score 方法。
    """

    def score(self, query: str, content: str) -> float:
        q = set(tokenize(query))
        d = set(tokenize(content))
        if not q or not d:
            return 0.0
        overlap = len(q & d) / len(q)
        phrase_bonus = 0.0
        if query.strip() and query.strip() in content:
            phrase_bonus = 0.35
        return min(1.0, overlap + phrase_bonus)

    def rerank(
            self,
            query,
            results,
            top_k,
            memory_type_hint=None,
            relation_hint=False,
            temporal_relation="at",
            expanded_query=None,
            predicate_hint=None,
            intent_hint=None,
    ):
        rescored = []
        t0 = time.perf_counter()

        for r in results:
            lexical = self.score(query, r["content"])
            expanded_lexical = self.score(
                expanded_query or query,
                r["content"],
            )
            lexical_signal = (
                    0.70 * lexical + 0.30 * expanded_lexical
            )

            base = float(r.get("score", 0.0))

            # 保留 RRF 分数的区分度，避免过早饱和。
            retrieval_signal = min(1.0, max(0.0, base * 20.0))

            mt = r.get("memory_type")
            metadata = r.get("metadata") or {}
            doc_predicate = metadata.get("predicate")

            structured = 0.0

            # 类型匹配
            if memory_type_hint and mt == memory_type_hint:
                structured += 0.15

            # 关系查询
            if relation_hint and mt == "relation":
                structured += 0.10

            # 时间查询
            if temporal_relation in ("before", "after"):
                if mt in ("fact", "event"):
                    structured += 0.05

            # 正负偏好方向
            if predicate_hint:
                if doc_predicate == predicate_hint:
                    structured += 0.15
                elif (
                        predicate_hint == "like"
                        and doc_predicate == "dislike"
                ):
                    structured -= 0.05
                elif (
                        predicate_hint == "dislike"
                        and doc_predicate == "like"
                ):
                    structured -= 0.05

            # 查询意图
            if intent_hint == "habit" and mt == "rule":
                structured += 0.15
            elif intent_hint == "event" and mt == "event":
                structured += 0.10
            elif intent_hint == "relation" and mt == "relation":
                structured += 0.10
            elif intent_hint == "fact" and mt == "fact":
                structured += 0.05

            # 当前有效事实
            if mt == "fact" and r.get("status") == "active":
                structured += 0.03

            structured = max(-0.1, min(0.5, structured))

            # 综合得分
            final = (
                    0.45 * lexical_signal
                    + 0.35 * retrieval_signal
                    + 0.20 * max(0.0, structured)
            )

            item = dict(r)
            item["score"] = round(final, 6)
            item["metadata"] = dict(metadata)
            item["metadata"]["rerank_score"] = round(
                lexical_signal, 6
            )
            item["metadata"]["expanded_rerank_score"] = round(
                expanded_lexical, 6
            )
            item["metadata"]["structured_bonus"] = round(
                structured, 6
            )
            item["metadata"]["retrieval_signal"] = round(
                retrieval_signal, 6
            )

            rescored.append(item)

        rescored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "RERANK_PROFILE input=%d output=%d elapsed=%.2f",
                len(results),
                min(len(rescored), top_k),
                (time.perf_counter() - t0) * 1000,
            )

        return rescored[:top_k]