from memory_engine.engine import MemoryEngine


def test_result_window_expands_immediate_same_session_neighbors_only():
    raws = [
        {"id": "m1", "session_id": "s1", "timestamp": 1, "role": "user", "content": "前置原因"},
        {"id": "m2", "session_id": "s1", "timestamp": 2, "role": "user", "content": "关键过程"},
        {"id": "m3", "session_id": "s1", "timestamp": 3, "role": "user", "content": "结果"},
        {"id": "m4", "session_id": "s1", "timestamp": 4, "role": "user", "content": "更远消息"},
        {"id": "m5", "session_id": "s2", "timestamp": 5, "role": "user", "content": "另一个 session"},
    ]
    candidates = [
        {
            "id": "m2",
            "content": "关键过程",
            "memory_type": "raw",
            "score": 1.0,
            "metadata": {"source_message_ids": ["m2"]},
        }
    ]

    expanded = MemoryEngine._expand_result_positions(candidates, raws, seed_k=20, window=1)
    expanded_ids = [item["id"] for item in expanded]

    assert expanded_ids[0] == "m2"
    assert "m1" in expanded_ids
    assert "m3" in expanded_ids
    assert "m4" not in expanded_ids
    assert "m5" not in expanded_ids

    neighbors = [item for item in expanded if item["id"] in {"m1", "m3"}]
    assert all(item["source"] == "result_window" for item in neighbors)
    assert all(item["metadata"]["center_message_id"] == "m2" for item in neighbors)
