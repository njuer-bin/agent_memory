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
Extract only durable information explicitly stated or strongly entailed by each message.
Do not invent facts, identities, dates, relationships, preferences, or causes.
Keep the original wording in content fields whenever possible.
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
