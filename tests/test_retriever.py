"""Verify the full retrieval pipeline: store → query → rank → access tracking."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from models.semantic import SemanticMemory
from stores.semantic_store import SemanticStore
from retrieval.retriever import UnifiedRetriever

TEST_DB = "./chroma_test_retriever"


def fresh_setup():
    shutil.rmtree(TEST_DB, ignore_errors=True)
    import config
    config.CHROMA_DB_PATH = TEST_DB
    store = SemanticStore()
    retriever = UnifiedRetriever(stores={"semantic": store})
    return store, retriever


def test_ranked_retrieval():
    store, retriever = fresh_setup()

    store.store(SemanticMemory(content="Python was created by Guido van Rossum"))
    store.store(SemanticMemory(content="The capital of France is Paris"))
    store.store(SemanticMemory(content="FastAPI uses Starlette under the hood"))

    results = retriever.query("Who created Python?", top_k=3)
    assert results[0].record.content == "Python was created by Guido van Rossum", (
        f"Expected Python fact first, got: {results[0].record.content}"
    )
    assert results[0].raw_similarity > 0.8, f"Expected similarity > 0.8, got {results[0].raw_similarity}"
    print(f"  PASS  correct fact ranks first (sim={results[0].raw_similarity:.4f})")


def test_access_count_increments():
    store, retriever = fresh_setup()

    store.store(SemanticMemory(content="Rust was created by Graydon Hoare"))

    for i in range(1, 4):
        results = retriever.query("Who created Rust?", top_k=1)
        record = store.get_by_id(results[0].record.id)
        assert record.access_count == i, f"Expected access_count={i}, got {record.access_count}"
    print(f"  PASS  access_count increments: 1 → 2 → 3")


def test_access_persists_across_instances():
    store, retriever = fresh_setup()
    store.store(SemanticMemory(content="Go was created at Google"))

    # Query twice
    retriever.query("Who created Go?", top_k=1)
    retriever.query("Who created Go?", top_k=1)

    # Create fresh store + retriever (simulates process restart)
    import config
    config.CHROMA_DB_PATH = TEST_DB
    store2 = SemanticStore()
    retriever2 = UnifiedRetriever(stores={"semantic": store2})

    results = retriever2.query("Who created Go?", top_k=1)
    record = store2.get_by_id(results[0].record.id)
    assert record.access_count == 3, f"Expected 3 (2 + 1 new), got {record.access_count}"
    print(f"  PASS  access_count persists across instances (count=3)")


def test_memory_types_filter():
    store, retriever = fresh_setup()
    store.store(SemanticMemory(content="Test fact"))

    results = retriever.query("test", memory_types=["semantic"])
    assert len(results) > 0, "Should find results when filtering for semantic"

    results = retriever.query("test", memory_types=["episodic"])
    assert len(results) == 0, "Should find nothing when filtering for episodic (no such store)"
    print(f"  PASS  memory_types filter works")


def cleanup():
    shutil.rmtree(TEST_DB, ignore_errors=True)


if __name__ == "__main__":
    print("Retriever tests:\n")
    try:
        test_ranked_retrieval()
        test_access_count_increments()
        test_access_persists_across_instances()
        test_memory_types_filter()
        print("\nAll tests passed.")
    finally:
        cleanup()
