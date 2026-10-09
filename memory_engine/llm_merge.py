from __future__ import annotations

from typing import Any

from .analyzer import Event, Fact, Profile, Relation, Rule
from .llm_analyzer import extract_batch
from .models import fingerprint, new_id
from .temporal_parser import normalize_temporal


def augment_analyzed(base: dict[str, Any], user_id: str, content: str, timestamp: int, source: str) -> dict[str, Any]:
    result = {key: list(base.get(key, [])) for key in ("facts", "relations", "events", "rules", "profiles")}
    payload = extract_batch([{"message_index": 0, "role": source, "content": content}])
    item = next(
        (x for x in payload.get("memories", []) or []
         if isinstance(x, dict) and str(x.get("message_index")) == "0"),
        None,
    )
    if not item:
        return base

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
        result["facts"].append(Fact(
            id=new_id("fact"), user_id=user_id, subject=subject,
            predicate=predicate, object=object_, content=text,
            timestamp=timestamp,
            fingerprint=fingerprint(user_id, "fact", subject, predicate, object_),
            valid_from=temporal.start if temporal.start is not None else timestamp,
            valid_to=temporal.end, temporal_text=temporal.text,
            source="gpt-4o-mini",
        ))

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
        result["relations"].append(Relation(
            id=new_id("rel"), user_id=user_id, subject=subject,
            predicate=predicate, object=object_, content=text,
            timestamp=timestamp,
            fingerprint=fingerprint(user_id, "rel", subject, predicate, object_),
        ))

    for row in item.get("events", []) or []:
        if not isinstance(row, dict):
            continue
        event = str(row.get("event") or "").strip()
        text = str(row.get("content") or content).strip()
        if event:
            result["events"].append(Event(
                id=new_id("event"), user_id=user_id, event=event,
                content=text, timestamp=timestamp,
                fingerprint=fingerprint(user_id, "event", event, text),
                event_start=temporal.start, event_end=temporal.end,
                temporal_text=temporal.text,
            ))

    for row in item.get("rules", []) or []:
        if not isinstance(row, dict):
            continue
        rule = str(row.get("rule") or "").strip()
        text = str(row.get("content") or content).strip()
        if rule:
            result["rules"].append(Rule(
                id=new_id("rule"), user_id=user_id, rule=rule,
                content=text, timestamp=timestamp,
                fingerprint=fingerprint(user_id, "rule", rule),
            ))

    for row in item.get("profiles", []) or []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "").strip()
        value = str(row.get("value") or "").strip()
        text = str(row.get("content") or content).strip()
        if key and value:
            result["profiles"].append(Profile(
                user_id=user_id, key=key, value=value,
                content=text, timestamp=timestamp,
            ))

    for key in ("facts", "relations", "events", "rules"):
        seen = {getattr(item, "fingerprint", None) for item in base.get(key, [])}
        result[key] = list(base.get(key, []))
        for item in result[key]:
            seen.add(getattr(item, "fingerprint", None))
        for item in result[key][len(base.get(key, [])):]:
            if getattr(item, "fingerprint", None) not in seen:
                seen.add(getattr(item, "fingerprint", None))
    # Deduplicate the merged structured memories by fingerprint.
    for key in ("facts", "relations", "events", "rules"):
        unique = []
        seen = set()
        for item in result[key]:
            fp = getattr(item, "fingerprint", None)
            if fp not in seen:
                seen.add(fp)
                unique.append(item)
        result[key] = unique
    result["profiles"] = list(base.get("profiles", [])) + result["profiles"]
    return result
