from __future__ import annotations


class EvidenceBuilder:
    def build(self, candidates, top_k):
        """Select evidence without dropping requirement coverage too early."""
        selected = []
        seen_content = set()
        used_ids = set()

        # First protect every evidence requirement. The previous implementation
        # capped each memory_type at three items, which could silently discard
        # the fourth/fifth independent evidence needed by LoCoMo multi-hop QA.
        requirement_ids = []
        for item in candidates:
            for req_id in item.get("_evidence_requirements", []) or []:
                if req_id not in requirement_ids:
                    requirement_ids.append(req_id)

        for req_id in requirement_ids:
            for item in candidates:
                if item.get("id") in used_ids:
                    continue
                if req_id not in (item.get("_evidence_requirements", []) or []):
                    continue
                content = str(item.get("content") or "").strip()
                if not content or content in seen_content:
                    continue
                selected.append(item)
                seen_content.add(content)
                used_ids.add(item.get("id"))
                break
            if len(selected) >= top_k:
                return selected[:top_k]

        # Then fill by global ranking. Keep the old anti-duplication behavior,
        # but do not impose a per-memory-type quota: evidence completeness is
        # more important than artificial type diversity for multi-hop queries.
        for item in candidates:
            if item.get("id") in used_ids:
                continue
            content = str(item.get("content") or "").strip()
            if not content or content in seen_content:
                continue
            selected.append(item)
            seen_content.add(content)
            used_ids.add(item.get("id"))
            if len(selected) >= top_k:
                break

        return selected[:top_k]
