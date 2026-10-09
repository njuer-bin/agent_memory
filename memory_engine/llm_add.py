from __future__ import annotations

import os
from typing import Any

from .analyzer import Event, Fact, Profile, Relation, Rule
from .llm_analyzer import extract_batch
from .models import fingerprint, new_id
from .temporal_parser import normalize_temporal


def _persist_extracted(engine, request, messages: list[dict[str, Any]], extracted: dict[str, Any]) -> None:
    by_index = {}
    for item in extracted.get("memories", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("message_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(messages):
            by_index[index] = item

    for index, message in enumerate(messages):
        item = by_index.get(index)
        if not item:
            continue
        timestamp = int(message["timestamp"])
        content = str(message["content"])
        temporal = normalize_temporal(content, timestamp)

        for row in item.get("facts", []) or []:
            if not isinstance(row, dict):
                continue
            subject = str(row.get("subject") or "user").strip()
            predicate = str(row.get("predicate") or "").strip()
            object_ = str(row.get("object") or "").strip()
            text = str(row.get("content") or content).strip()
            if not predicate or not object_:
                continue
            if subject in {"我", "我的", "用户"}:
                subject = "user"
            fact = Fact(
                id=new_id("fact"), user_id=request.user_id, subject=subject,
                predicate=predicate, object=object_, content=text,
                timestamp=timestamp,
                fingerprint=fingerprint(request.user_id, "fact", subject, predicate, object_),
                valid_from=temporal.start if temporal.start is not None else timestamp,
                valid_to=temporal.end, temporal_text=temporal.text,
                source="gpt-4o-mini",
            )
            inserted, _ = engine.governance.accept_fact(fact)
            if inserted:
                vector = engine.embedder.embed(fact.content)
                engine.store.embed(fact.id, request.user_id, vector)
                engine.vector_index.add(request.user_id, fact.id, vector)

        for row in item.get("relations", []) or []:
            if not isinstance(row, dict):
                continue
            subject = str(row.get("subject") or "").strip()
            predicate = str(row.get("predicate") or "").strip()
            object_ = str(row.get("object") or "").strip()
            text = str(row.get("content") or content).strip()
            if not subject or not predicate or not object_ or subject == object_:
                continue
            if subject in {"我", "我的", "用户"}:
                subject = "user"
            if object_ in {"我", "我的", "用户"}:
                object_ = "user"
            rel = Relation(
                id=new_id("rel"), user_id=request.user_id, subject=subject,
                predicate=predicate, object=object_, content=text,
                timestamp=timestamp,
                fingerprint=fingerprint(request.user_id, "rel", subject, predicate, object_),
            )
            engine.store.insert_relation(rel)
            vector = engine.embedder.embed(rel.content)
            engine.store.embed(rel.id, request.user_id, vector)
            engine.vector_index.add(request.user_id, rel.id, vector)

        for row in item.get("events", []) or []:
            if not isinstance(row, dict):
                continue
            event = str(row.get("event") or "").strip()
            text = str(row.get("content") or content).strip()
            if not event:
                continue
            obj = Event(
                id=new_id("event"), user_id=request.user_id, event=event,
                content=text, timestamp=timestamp,
                fingerprint=fingerprint(request.user_id, "event", event, text),
                event_start=temporal.start, event_end=temporal.end,
                temporal_text=temporal.text,
            )
            engine.store.insert_event(obj)
            vector = engine.embedder.embed(obj.content)
            engine.store.embed(obj.id, request.user_id, vector)
            engine.vector_index.add(request.user_id, obj.id, vector)

        for row in item.get("rules", []) or []:
            if not isinstance(row, dict):
                continue
            rule = str(row.get("rule") or "").strip()
            text = str(row.get("content") or content).strip()
            if not rule:
                continue
            obj = Rule(
                id=new_id("rule"), user_id=request.user_id, rule=rule,
                content=text, timestamp=timestamp,
                fingerprint=fingerprint(request.user_id, "rule", rule),
            )
            engine.store.insert_rule(obj)
            vector = engine.embedder.embed(obj.content)
            engine.store.embed(obj.id, request.user_id, vector)
            engine.vector_index.add(request.user_id, obj.id, vector)

        for row in item.get("profiles", []) or []:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip()
            value = str(row.get("value") or "").strip()
            text = str(row.get("content") or content).strip()
            if key and value:
                engine.store.upsert_profile(Profile(
                    user_id=request.user_id, key=key, value=value,
                    content=text, timestamp=timestamp,
                ))


def add_with_llm(engine, request) -> bool:
    """AML-compliant Add: claim -> gpt-4o-mini -> deterministic engine -> persist LLM memories."""
    if not engine.store.claim_request(request.request_id, request.user_id):
        return True

    try:
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

        extracted_batches = []
        for batch in batches:
            messages = [
                {"message_index": i, "role": msg.role, "content": msg.content, "timestamp": msg.timestamp or 0}
                for i, msg in enumerate(batch)
            ]
            extracted_batches.append((messages, extract_batch(messages)))

        # Reuse the existing canonical raw-history and deterministic storage path.
        engine._add_claimed(request)

        for messages, extracted in extracted_batches:
            for message in messages:
                if not message["timestamp"]:
                    message["timestamp"] = 0
            _persist_extracted(engine, request, messages, extracted)
        return True
    except Exception:
        engine.store.release_request(request.request_id)
        raise
