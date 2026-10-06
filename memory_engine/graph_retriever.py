from __future__ import annotations


class GraphRetriever:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _node_keys(rel: dict) -> set[str]:
        return {
            str(rel.get("subject", "")).strip(),
            str(rel.get("object", "")).strip(),
        } - {""}

    @staticmethod
    def _as_result(user_id: str, rel: dict, score: float, hop: int, path: list[str]) -> dict:
        return {
            "id": rel["id"],
            "content": rel["content"],
            "role": "relation",
            "timestamp": rel["timestamp"],
            "user_id": user_id,
            "session_id": "",
            "score": score,
            "source": "graph_expansion",
            "memory_type": "relation",
            "status": "active",
            "valid_from": rel["timestamp"],
            "valid_to": None,
            "metadata": {
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],
                "graph_hop": hop,
                "graph_path": path,
            },
        }

    def expand(self, user_id: str, seed_results: list[dict], limit=10, max_hops=2):
        """Expand relation evidence by up to two hops.

        P1 only matched relations directly touching a seed. That is insufficient
        for questions whose answer requires A -> B -> C. P2 performs a small,
        bounded BFS over the user's relation graph and keeps the path in
        metadata so downstream ranking/evidence reconstruction can distinguish
        direct from second-hop evidence.
        """
        rels = self.store.relations(user_id)
        if not rels or not seed_results:
            return []

        # Seed only structured entities; raw sentence text is too broad and can
        # accidentally turn unrelated lexical matches into graph expansion.
        frontier: set[str] = set()
        for item in seed_results:
            md = item.get("metadata", {}) or {}
            for key in ("subject", "object", "value"):
                value = md.get(key)
                if value:
                    frontier.add(str(value).strip())

        if not frontier:
            return []

        seen_relations: set[str] = set()
        seen_nodes = set(frontier)
        out: list[dict] = []
        paths: dict[str, list[str]] = {node: [node] for node in frontier}

        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for rel in rels:
                rid = rel["id"]
                if rid in seen_relations:
                    continue
                nodes = self._node_keys(rel)
                if not (nodes & frontier):
                    continue

                seen_relations.add(rid)
                subject = str(rel["subject"])
                obj = str(rel["object"])
                anchor = next(iter(nodes & frontier))
                base_path = paths.get(anchor, [anchor])

                if subject == anchor:
                    other = obj
                elif obj == anchor:
                    other = subject
                else:
                    continue

                path = base_path + [other]
                score = 0.018 if hop == 1 else 0.010
                out.append(self._as_result(user_id, rel, score, hop, path))

                if other not in seen_nodes:
                    seen_nodes.add(other)
                    paths[other] = path
                    next_frontier.add(other)

                if len(out) >= limit:
                    return out

            frontier = next_frontier
            if not frontier:
                break

        return out[:limit]
