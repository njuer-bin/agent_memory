from memory_engine.p12_long_history import build_evidence_dossier, is_global_history_query


def _raw(rid, session, ts, content):
    return {
        "id": rid,
        "user_id": "u1",
        "session_id": session,
        "role": "user",
        "timestamp": ts,
        "content": content,
    }


def test_global_history_query_detection():
    assert is_global_history_query("总结一下我过去的经历")
    assert is_global_history_query("What do you remember about me?")
    assert not is_global_history_query("我现在住在哪里？")


def test_dossier_preserves_sessions_dates_and_raw_text():
    raws = [
        _raw("a1", "s1", 1000, "我住在杭州。"),
        _raw("a2", "s1", 2000, "后来我开始跑步。"),
        _raw("b1", "s2", 3000, "我搬到了上海。"),
        _raw("b2", "s2", 4000, "我准备参加马拉松。"),
    ]
    dossier = build_evidence_dossier("总结我的历史", raws)
    assert dossier is not None
    assert dossier["source"] == "p12_history_dossier"
    assert dossier["metadata"]["lossless"] is True
    assert set(dossier["metadata"]["session_ids"]) == {"s1", "s2"}
    content = dossier["content"]
    assert "session=s1" in content
    assert "session=s2" in content
    assert "timestamp=1000" in content or "timestamp=2000" in content
    assert "timestamp=3000" in content or "timestamp=4000" in content
    assert "我住在杭州。" in content
    assert "我搬到了上海。" in content


def test_dossier_is_not_created_for_point_fact_query():
    raws = [_raw("a1", "s1", 1000, "我住在杭州。")]
    assert build_evidence_dossier("我现在住在哪里？", raws) is None
