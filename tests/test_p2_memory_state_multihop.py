from memory_engine.graph_retriever import GraphRetriever


class FakeStore:
    def relations(self, user_id):
        return [
            {
                "id": "r1",
                "subject": "user",
                "predicate": "friend",
                "object": "Bob",
                "content": "用户的朋友是 Bob",
                "timestamp": 1,
            },
            {
                "id": "r2",
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Acme",
                "content": "Bob 在 Acme 工作",
                "timestamp": 2,
            },
            {
                "id": "r3",
                "subject": "Carol",
                "predicate": "works_at",
                "object": "Other",
                "content": "Carol 在 Other 工作",
                "timestamp": 3,
            },
        ]


def test_graph_retriever_expands_two_hops():
    retriever = GraphRetriever(FakeStore())
    seeds = [{
        "metadata": {"subject": "user", "object": "Bob"},
        "content": "用户的朋友是 Bob",
    }]

    results = retriever.expand("u1", seeds, limit=10, max_hops=2)
    ids = [item["id"] for item in results]

    assert "r1" in ids
    assert "r2" in ids
    assert "r3" not in ids

    second_hop = next(item for item in results if item["id"] == "r2")
    assert second_hop["metadata"]["graph_hop"] == 2
    assert second_hop["metadata"]["graph_path"] == ["user", "Bob", "Acme"]
