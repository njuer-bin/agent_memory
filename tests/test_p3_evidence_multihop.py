from memory_engine.analyzer import MemoryAnalyzer
from memory_engine.multihop_evidence import EvidenceAnchorExtractor


def test_raw_memory_extracts_entity_anchor():
    extractor = EvidenceAnchorExtractor(MemoryAnalyzer())
    item = {
        "user_id": "u1",
        "timestamp": 1,
        "content": "我的朋友是 Bob",
        "metadata": {},
    }
    anchors = extractor.extract(item)
    assert "Bob" in anchors


def test_graph_anchor_annotation_preserves_bridge():
    extractor = EvidenceAnchorExtractor(MemoryAnalyzer())
    item = {"content": "Bob 在 Acme 工作", "metadata": {}}
    annotated = extractor.annotate(item, ["Bob", "Acme"])
    assert annotated["metadata"]["evidence_anchors"] == ["Bob", "Acme"]
