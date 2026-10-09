from __future__ import annotations

import os
import uuid

import requests


BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
USER = "llm-test-" + uuid.uuid4().hex[:8]
SESSION = "llm-session-" + uuid.uuid4().hex[:8]
RID = "llm-" + uuid.uuid4().hex



def post(path: str, payload: dict) -> dict:
    response = requests.post(BASE + path, json=payload, timeout=60)
    print(f"\n{path} {response.status_code}")
    print(response.text)
    response.raise_for_status()
    return response.json()



def search(query: str) -> list[dict]:
    result = post("/search", {
        "query": query,
        "user_id": USER,
        "top_k": 10,
    })
    data = result.get("data", [])
    assert isinstance(data, list), f"search data is not a list: {data!r}"
    return data



def contents(rows: list[dict]) -> str:
    return "\n".join(str(row.get("content", "")) for row in rows)



def assert_contains(query: str, expected: list[str]) -> list[dict]:
    rows = search(query)
    text = contents(rows)
    missing = [item for item in expected if item not in text]
    print(f"\n>>> {query}")
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row.get('content')} (score={row.get('score')})")
    assert not missing, f"Missing evidence for {query!r}: {missing}. Retrieved: {text}"
    return rows



def main() -> None:
    print("=== Strict LLM Add Test ===")
    print("user_id:", USER)
    print("request_id:", RID)

    payload = {
        "request_id": RID,
        "messages": [{
            "role": "user",
            "timestamp": 1735689600000,
            "content": (
                "我叫小明，目前在杭州工作。"
                "去年我从北京搬到了杭州，因为杭州离我的公司更近。"
                "我平时喜欢喝咖啡，不喜欢早班飞机。"
            ),
        }],
        "user_id": USER,
        "session_id": SESSION,
    }

    result = post("/add", payload)
    assert result["success"] is True
    assert result["request_id"] == RID

    # Evidence completeness: every independently useful fact/relation/preference
    # should be represented somewhere in the Top-10, not only hidden in the raw message.
    assert_contains("小明在哪里工作？", ["杭州"])
    assert_contains("小明从哪里搬到了哪里？", ["北京", "杭州"])
    assert_contains("小明为什么搬到杭州？", ["公司", "更近"])
    assert_contains("小明喜欢喝什么？", ["咖啡"])
    assert_contains("小明喜欢早班飞机吗？", ["不喜欢", "早班飞机"])

    print("\n=== STRICT LLM ADD TEST PASSED ===")


if __name__ == "__main__":
    main()
