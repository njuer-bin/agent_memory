from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


MODEL = os.getenv("AML_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = float(os.getenv("AML_OPENAI_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("AML_OPENAI_RETRIES", "2"))

SYSTEM_PROMPT = """
You are the memory-writing component of a personal-memory system.
Extract durable information explicitly stated or strongly entailed by each message.
Do not invent facts, identities, dates, relationships, preferences, or causes.
Preserve the user's meaning and keep the original wording in content fields whenever possible.

The goal is NOT to summarize the whole message. Decompose a message into small, independently
retrievable memory units. One message can and often should produce multiple memories.

Important extraction rules:
1. Facts: stable attributes or explicit states, such as name, residence, workplace, role, or preference
   when it is naturally represented as a fact.
2. Relations: explicit relationships between entities or between an event and its cause. This is
   especially important for multi-hop questions. If the message says "I moved from Beijing to
   Hangzhou because Hangzhou is closer to my company", extract separate relations for the move and
   the reason, for example:
   - subject=user/person, predicate=moved_from, object=Beijing
   - subject=user/person, predicate=moved_to, object=Hangzhou
   - subject=move/event, predicate=reason, object=Hangzhou is closer to my company
   Use the actual person/entity name when explicitly available instead of inventing "user".
3. Events: concrete actions or changes that happened, including moving from one place to another.
   Keep the event content focused on that event rather than copying the entire message.
4. Profiles: durable likes, dislikes, habits, preferences, or other user traits. For example,
   "I like coffee" should become key="likes", value="coffee", content="I like coffee" and
   "I dislike early flights" should become key="dislikes", value="early flights", content="I dislike early flights".
5. Rules: only use for explicit if/then instructions or durable behavioral rules. Do not turn an
   ordinary preference into a rule unless that is the clearest representation.
6. Causality: never drop an explicit reason introduced by because, since, due to, therefore, etc.
   A causal statement must produce a retrievable relation whose predicate expresses the causal link.
7. Negation: preserve explicit negation such as "不喜欢". Never turn a dislike into a like.
8. Temporal words such as "去年" should remain in content; do not invent an exact date.
9. Avoid duplicate whole-message copies. Content should be the smallest useful evidence span.

Return JSON only with this schema:
{
  "memories": [
    {
      "message_index": 0,
      "facts": [{"subject":"user|person|entity", "predicate":"...", "object":"...", "content":"..."}],
      "relations": [{"subject":"...", "predicate":"...", "object":"...", "content":"..."}],
      "events": [{"event":"...", "content":"..."}],
      "rules": [{"rule":"...", "content":"..."}],
      "profiles": [{"key":"...", "value":"...", "content":"..."}]
    }
  ]
}
Use empty arrays when a category is not present. A message can have multiple memories.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("OpenAI-compatible memory extractor returned no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("OpenAI-compatible memory extractor returned non-object JSON")
    return value


def extract_batch(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract structured memory with the configured OpenAI-compatible endpoint."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    required = os.getenv("AML_OPENAI_REQUIRED", "1").strip().lower() not in {"0", "false", "no"}
    if not api_key:
        if required:
            raise RuntimeError("OPENAI_API_KEY is required for AML Add (gpt-4o-mini)")
        return {"memories": []}

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenAI-compatible Base URL. For UIUI use https://api.uiuihao.com/v1.
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.uiuihao.com/v1").rstrip("/")
    url = base_url + "/chat/completions"

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 4))
                continue
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            return _extract_json(text)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 4))
                continue
            break
    raise RuntimeError(f"gpt-4o-mini memory extraction failed: {last_error}") from last_error
