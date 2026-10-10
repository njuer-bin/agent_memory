from __future__ import annotations


class EvidenceBuilder:
    """Select diverse evidence while protecting requirement coverage.

    Retrieval exposes several views of the same raw turn (raw/window/session,
    provenance, and adjacent-message pairs). Pair windows are explicit
    multi-message evidence and must remain eligible even when their source
    messages also appear as individual raw results.
    """

    PRIMARY_TYPES = {"raw", "fact", "event", "relation", "rule", "profile"}
    CONTEXT_TYPES = {"window", "session"}

    @classmethod
    def _kind_rank(cls, item):
        source = str(item.get("source") or "")
        memory_type = str(item.get("memory_type") or "")
        # Pair windows can be the only result that contains the complete
        # cross-turn evidence chain, so don't let individual raw rows crowd
        # them out of a small final top-k.
        if source == "result_window_pair":
            return -1
        if source == "evidence_provenance":
            return 3
        if memory_type in cls.PRIMARY_TYPES:
            return 0
        if memory_type in cls.CONTEXT_TYPES:
            return 1
        return 2

    @staticmethod
    def _source_ids(item):
        md = item.get("metadata", {}) or {}
        ids = md.get("source_message_ids") or []
        if ids:
            return set(str(x) for x in ids if x)
        if item.get("memory_type") == "raw" and item.get("id"):
            return {str(item["id"])}
        return set()

    @staticmethod
    def _may_overlap_selected_sources(item):
        # Unlike a duplicate contextual wrapper, a pair window intentionally
        # combines two original messages into one complete evidence unit.
        return str(item.get("source") or "") == "result_window_pair"

    def build(self, candidates, top_k):
        """Select evidence without dropping requirement coverage too early."""
        selected = []
        seen_content = set()
        used_ids = set()
        selected_source_ids = set()

        requirement_ids = []
        for item in candidates:
            for req_id in item.get("_evidence_requirements", []) or []:
                if req_id not in requirement_ids:
                    requirement_ids.append(req_id)

        # First protect every requirement, but prefer a primary/message-level
        # candidate over a contextual window or provenance wrapper.
        for req_id in requirement_ids:
            eligible = [
                item for item in candidates
                if item.get("id") not in used_ids
                and req_id in (item.get("_evidence_requirements", []) or [])
                and str(item.get("content") or "").strip()
            ]
            eligible.sort(key=lambda item: (
                self._kind_rank(item),
                -float(item.get("_requirement_score", item.get("score", 0.0))),
                -float(item.get("score", 0.0)),
            ))
            chosen = None
            for item in eligible:
                source_ids = self._source_ids(item)
                if (
                    self._kind_rank(item) > 0
                    and not self._may_overlap_selected_sources(item)
                    and source_ids & selected_source_ids
                ):
                    continue
                chosen = item
                break
            if chosen is not None:
                selected.append(chosen)
                content = str(chosen.get("content") or "").strip()
                seen_content.add(content)
                used_ids.add(chosen.get("id"))
                selected_source_ids.update(self._source_ids(chosen))
            if len(selected) >= top_k:
                return selected[:top_k]

        # Global fill: complete pair windows rank ahead of individual rows;
        # ordinary context views are still suppressed when redundant.
        remaining = [item for item in candidates if item.get("id") not in used_ids]
        remaining.sort(key=lambda item: (
            self._kind_rank(item),
            -float(item.get("score", 0.0)),
        ))
        for item in remaining:
            content = str(item.get("content") or "").strip()
            if not content or content in seen_content:
                continue
            source_ids = self._source_ids(item)
            if (
                self._kind_rank(item) > 0
                and not self._may_overlap_selected_sources(item)
                and source_ids & selected_source_ids
            ):
                continue
            selected.append(item)
            seen_content.add(content)
            used_ids.add(item.get("id"))
            selected_source_ids.update(source_ids)
            if len(selected) >= top_k:
                break

        return selected[:top_k]
