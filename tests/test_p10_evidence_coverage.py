from memory_engine.evidence import EvidenceBuilder
from memory_engine.query_analyzer import QueryAnalyzer


def test_answer_shape_requirements_for_location_question():
    plan = QueryAnalyzer().analyze("Where has Maria made friends?", forced_multi_hop=True)
    requirements = QueryAnalyzer.evidence_requirements(plan)

    queries = [r["query"].lower() for r in requirements]
    assert any(r["kind"] == "location" for r in requirements)
    assert any("maria" in q and "location" in q for q in queries)


def test_answer_shape_requirements_for_look_like_question():
    plan = QueryAnalyzer().analyze(
        "What Jon thinks the ideal dance studio should look like?",
        forced_multi_hop=True,
    )
    requirements = QueryAnalyzer.evidence_requirements(plan)

    assert any(r["kind"] == "attribute" for r in requirements)
    assert any("jon" in r["query"].lower() and "characteristics" in r["query"].lower()
               for r in requirements)


def test_evidence_builder_prefers_raw_over_context_duplicates():
    raw = {
        "id": "raw-1",
        "content": "Jon wants Marley flooring.",
        "memory_type": "raw",
        "source": "raw",
        "score": 0.50,
        "_evidence_requirements": ["R1"],
        "metadata": {"source_message_ids": ["raw-1"]},
    }
    window = {
        "id": "window-1",
        "content": "context around Jon and Marley flooring",
        "memory_type": "window",
        "source": "session_window",
        "score": 0.90,
        "_evidence_requirements": ["R1"],
        "metadata": {"source_message_ids": ["raw-1", "raw-2"]},
    }
    provenance = {
        "id": "prov-1",
        "content": "Jon wants Marley flooring.",
        "memory_type": "raw",
        "source": "evidence_provenance",
        "score": 0.95,
        "_evidence_requirements": ["R1"],
        "metadata": {"source_message_ids": ["raw-1"]},
    }

    selected = EvidenceBuilder().build([window, provenance, raw], top_k=3)

    assert selected[0]["id"] == "raw-1"
    assert all(item["id"] != "window-1" for item in selected[1:])


def test_evidence_builder_keeps_independent_raw_requirements():
    raw1 = {
        "id": "raw-1",
        "content": "John did taekwondo.",
        "memory_type": "raw",
        "source": "raw",
        "score": 0.60,
        "_evidence_requirements": ["R1"],
        "metadata": {"source_message_ids": ["raw-1"]},
    }
    raw2 = {
        "id": "raw-2",
        "content": "John did kickboxing.",
        "memory_type": "raw",
        "source": "raw",
        "score": 0.55,
        "_evidence_requirements": ["R2"],
        "metadata": {"source_message_ids": ["raw-2"]},
    }

    selected = EvidenceBuilder().build([raw1, raw2], top_k=2)

    assert [item["id"] for item in selected] == ["raw-1", "raw-2"]
