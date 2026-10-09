from __future__ import annotations

from memory_engine.llm_analyzer import extract_batch


MESSAGE = {
    "message_index": 0,
    "role": "user",
    "content": (
        "我叫小明，目前在杭州工作。"
        "去年我从北京搬到了杭州，因为杭州离我的公司更近。"
        "我平时喜欢喝咖啡，不喜欢早班飞机。"
    ),
    "timestamp": 1735689600000,
}


def main() -> None:
    print("=== Direct gpt-4o-mini Extraction Test ===")
    result = extract_batch([MESSAGE])
    print(result)

    memories = result.get("memories", [])
    assert memories, "LLM returned no memories"

    item = memories[0]
    facts = item.get("facts", []) or []
    relations = item.get("relations", []) or []
    events = item.get("events", []) or []
    profiles = item.get("profiles", []) or []

    text = str(result)
    checks = {
        "name": "小明" in text,
        "work_location": "杭州" in text,
        "move_from": "北京" in text,
        "move_to": "杭州" in text,
        "cause": ("公司" in text and "更近" in text),
        "like_coffee": "咖啡" in text,
        "dislike_early_flight": ("不喜欢" in text and "早班飞机" in text),
    }

    print("\n=== Category Counts ===")
    print("facts:", len(facts))
    print("relations:", len(relations))
    print("events:", len(events))
    print("profiles:", len(profiles))
    print("\n=== Required Evidence ===")
    for key, ok in checks.items():
        print(f"{key}: {'OK' if ok else 'MISSING'}")

    missing = [key for key, ok in checks.items() if not ok]
    assert not missing, f"LLM extraction missing evidence: {missing}"

    print("\n=== DIRECT LLM EXTRACTION PASSED ===")


if __name__ == "__main__":
    main()
