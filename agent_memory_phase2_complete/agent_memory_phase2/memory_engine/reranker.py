from __future__ import annotations

from .store import tokenize


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

    def rerank(self, query, results, top_k, memory_type_hint=None, relation_hint=False,
               temporal_relation="at", expanded_query=None, predicate_hint=None, intent_hint=None):
        rescored = []
        for r in results:
            lexical = self.score(query, r["content"])
            expanded_lexical = self.score(expanded_query or query, r["content"])
            lexical_signal = 0.70 * lexical + 0.30 * expanded_lexical
            base = float(r.get("score", 0.0))
            structured = 0.0
            if memory_type_hint and r.get("memory_type") == memory_type_hint:
                structured += 0.10
            if relation_hint and r.get("memory_type") == "relation":
                structured += 0.08
            if temporal_relation in ("before", "after") and r.get("memory_type") in ("fact", "event"):
                structured += 0.03
            if predicate_hint and r.get("metadata", {}).get("predicate") == predicate_hint:
                structured += 0.08
            if intent_hint == "habit" and r.get("memory_type") == "rule":
                structured += 0.08
            final = 0.50 * lexical_signal + 0.35 * min(1.0, base * 60.0) + 0.15 * structured
            item = dict(r)
            item["score"] = round(final, 6)
            item["metadata"] = dict(item.get("metadata", {}))
            item["metadata"]["rerank_score"] = round(lexical_signal, 6)
            item["metadata"]["expanded_rerank_score"] = round(expanded_lexical, 6)
            item["metadata"]["structured_bonus"] = round(structured, 6)
            rescored.append(item)
        rescored.sort(key=lambda x:x["score"], reverse=True)
        return rescored[:top_k]
