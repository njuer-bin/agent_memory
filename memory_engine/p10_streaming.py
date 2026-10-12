"""Small P10 streaming/internal-transfer layer; B2 is intentionally untouched."""

from __future__ import annotations

from .models import new_id

_TRANSFER_MARKERS = ("关联", "联系", "迁移", "延续", "之前提到", "沿用", "相关记忆", "related", "linked", "transfer", "carry over", "continue", "previously mentioned")


def _has(query: str) -> bool:
    q = (query or "").lower()
    return any(x.lower() in q for x in _TRANSFER_MARKERS)


def _linked_rows(engine, request) -> list[dict]:
    if not _has(request.query):
        return []
    links = engine.store.memory_links(request.user_id)
    if not links:
        return []
    ids = []
    for link in links[:24]:
        for key in ("source_id", "target_id"):
            value = str(link.get(key) or "").strip()
            if value and value not in ids:
                ids.append(value)
    raws = {str(x.get("id")): x for x in engine.store.raw_by_ids(request.user_id, ids)}
    out = []
    for link in links[:16]:
        source = str(link.get("source_id") or "")
        target = str(link.get("target_id") or "")
        for mid in (source, target):
            raw = raws.get(mid)
            if not raw:
                continue
            out.append({
                "id": f"p10-link:{link.get('id')}:{mid}",
                "content": raw.get("content", ""),
                "role": raw.get("role", "memory"),
                "timestamp": raw.get("timestamp", 0),
                "user_id": request.user_id,
                "session_id": raw.get("session_id", ""),
                "score": 0.80,
                "source": "p10_link_transfer",
                "memory_type": "raw",
                "status": "active",
                "valid_from": raw.get("timestamp", 0),
                "valid_to": None,
                "metadata": {
                    "p10_channel": "internal_transfer",
                    "link_id": link.get("id"),
                    "link_relation": link.get("relation"),
                    "link_confidence": link.get("confidence"),
                    "source_message_ids": [mid],
                },
            })
    return out


def _install() -> None:
    from . import engine as engine_module
    if getattr(engine_module.MemoryEngine, "_p10_streaming_installed", False):
        return
    previous_search = engine_module.MemoryEngine.search

    def search_with_links(self, request):
        rows = list(previous_search(self, request))
        query = (request.query or request.question or "").strip()
        # Do not enter the causal/B2 path at all.
        if getattr(request, "multi_hop", False) and any(x in query.lower() for x in ("为什么", "为何", "原因", "导致", "why", "cause", "because")):
            return rows
        try:
            seen = {str(x.get("id")) for x in rows}
            for item in _linked_rows(self, request):
                if item["id"] not in seen:
                    rows.append(item)
                    seen.add(item["id"])
            rows.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
            return rows[: request.top_k]
        except Exception:
            return rows[: request.top_k]

    engine_module.MemoryEngine.search = search_with_links
    engine_module.MemoryEngine._p10_streaming_installed = True


_install()
