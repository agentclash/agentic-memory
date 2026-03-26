"""Verify the full retrieval pipeline: store → query → rank → access tracking."""

import sys
import os
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import config
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from stores.episodic_store import EpisodicStore
from stores.semantic_store import SemanticStore
from retrieval.retriever import UnifiedRetriever
from tests.helpers import HashingEmbedder

_DB_PATHS = []


def fresh_setup():
    db_path = tempfile.mkdtemp(prefix="chroma_test_retriever_")
    _DB_PATHS.append(db_path)
    shutil.rmtree(db_path, ignore_errors=True)
    config.CHROMA_DB_PATH = db_path
    store = SemanticStore(embedder=HashingEmbedder())
    retriever = UnifiedRetriever(stores={"semantic": store})
    return store, retriever, db_path


def fresh_mixed_setup():
    db_path = tempfile.mkdtemp(prefix="chroma_test_retriever_mixed_")
    _DB_PATHS.append(db_path)
    shutil.rmtree(db_path, ignore_errors=True)
    config.CHROMA_DB_PATH = db_path
    semantic_store = SemanticStore(embedder=HashingEmbedder())
    episodic_store = EpisodicStore(embedder=HashingEmbedder())
    retriever = UnifiedRetriever(stores={"semantic": semantic_store, "episodic": episodic_store})
    return semantic_store, episodic_store, retriever, db_path


def test_ranked_retrieval():
    store, retriever, _ = fresh_setup()

    store.store(SemanticMemory(content="Python was created by Guido van Rossum"))
    store.store(SemanticMemory(content="The capital of France is Paris"))
    store.store(SemanticMemory(content="FastAPI uses Starlette under the hood"))

    results = retriever.query("Who created Python?", top_k=3)
    assert results[0].record.content == "Python was created by Guido van Rossum", (
        f"Expected Python fact first, got: {results[0].record.content}"
    )
    assert results[0].raw_similarity > results[1].raw_similarity, (
        "Expected the best match to score above the runner-up"
    )
    print(f"  PASS  correct fact ranks first (sim={results[0].raw_similarity:.4f})")


def test_access_count_increments():
    store, retriever, _ = fresh_setup()

    store.store(SemanticMemory(content="Rust was created by Graydon Hoare"))

    for i in range(1, 4):
        results = retriever.query("Who created Rust?", top_k=1)
        # Returned record should reflect the updated count (not off-by-one)
        assert results[0].record.access_count == i, (
            f"Expected returned access_count={i}, got {results[0].record.access_count}"
        )
        # Persisted count should match
        record = store.get_by_id(results[0].record.id)
        assert record.access_count == i, f"Expected persisted access_count={i}, got {record.access_count}"
    print(f"  PASS  access_count increments: 1 → 2 → 3 (returned + persisted match)")


def test_access_persists_across_instances():
    store, retriever, db_path = fresh_setup()
    store.store(SemanticMemory(content="Go was created at Google"))

    # Query twice
    retriever.query("Who created Go?", top_k=1)
    retriever.query("Who created Go?", top_k=1)

    # Create fresh store + retriever (simulates process restart)
    import config
    config.CHROMA_DB_PATH = db_path
    store2 = SemanticStore(embedder=HashingEmbedder())
    retriever2 = UnifiedRetriever(stores={"semantic": store2})

    results = retriever2.query("Who created Go?", top_k=1)
    record = store2.get_by_id(results[0].record.id)
    assert record.access_count == 3, f"Expected 3 (2 + 1 new), got {record.access_count}"
    print(f"  PASS  access_count persists across instances (count=3)")


def test_memory_types_filter():
    store, retriever, _ = fresh_setup()
    store.store(SemanticMemory(content="Test fact"))

    results = retriever.query("test", memory_types=["semantic"])
    assert len(results) > 0, "Should find results when filtering for semantic"

    results = retriever.query("test", memory_types=["episodic"])
    assert len(results) == 0, "Should find nothing when filtering for episodic (no such store)"
    print(f"  PASS  memory_types filter works")


def test_over_fetch():
    """Verify that the reranker sees more candidates than top_k."""
    store, retriever, _ = fresh_setup()

    store.store(SemanticMemory(content="Alpha fact about testing"))
    store.store(SemanticMemory(content="Beta fact about testing"))
    store.store(SemanticMemory(content="Gamma fact about testing"))

    # With top_k=1 and over-fetch, the reranker still gets 3 candidates
    results = retriever.query("testing", top_k=1)
    assert len(results) == 1, "Should return exactly top_k results"
    print(f"  PASS  top_k=1 returns 1 result (reranker had 3 candidates)")


def test_mixed_store_retrieval():
    semantic_store, episodic_store, retriever, _ = fresh_mixed_setup()
    now = datetime.now(timezone.utc)

    semantic_store.store(
        SemanticMemory(
            content="The retrieval engine indexes documents and facts",
            created_at=now,
            importance=0.6,
        )
    )
    episodic_store.store(
        EpisodicMemory(
            content="We debugged the retrieval engine during a session",
            session_id="session-1",
            created_at=now,
            importance=0.6,
        )
    )

    results = retriever.query("retrieval engine", top_k=2)

    assert len(results) == 2
    assert {result.record.memory_type for result in results} == {"semantic", "episodic"}
    print("  PASS  mixed retrieval returns semantic and episodic records together")


def test_query_recent_and_time_range_return_episodic_only():
    semantic_store, episodic_store, retriever, _ = fresh_mixed_setup()
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    semantic_store.store(
        SemanticMemory(
            content="A semantic fact that should not appear in temporal queries",
            created_at=base + timedelta(minutes=30),
        )
    )
    episodic_store.store(
        EpisodicMemory(
            content="Earlier episode",
            session_id="session-1",
            created_at=base,
        )
    )
    episodic_store.store(
        EpisodicMemory(
            content="Later episode",
            session_id="session-2",
            created_at=base + timedelta(minutes=10),
        )
    )

    recent = retriever.query_recent(1)
    ranged = retriever.query_time_range(base, base + timedelta(minutes=10))

    assert [record.content for record in recent] == ["Later episode"]
    assert [record.memory_type for record in ranged] == ["episodic", "episodic"]
    assert [record.content for record in ranged] == ["Earlier episode", "Later episode"]
    print("  PASS  direct temporal queries return episodic records only")


def test_temporal_queries_update_episodic_access_counts():
    _, episodic_store, retriever, _ = fresh_mixed_setup()
    record_id = episodic_store.store(
        EpisodicMemory(
            content="Access-tracked episode",
            session_id="session-access",
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    )

    recent = retriever.query_recent(1)
    ranged = retriever.query_time_range(
        datetime(2026, 1, 1, 11, 59, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
    )
    persisted = episodic_store.get_by_id(record_id)

    assert recent[0].access_count == 1
    assert ranged[0].access_count == 2
    assert persisted.access_count == 2
    print("  PASS  temporal retriever queries update episodic access tracking")


def cleanup():
    for d in _DB_PATHS:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    print("Retriever tests:\n")
    try:
        test_ranked_retrieval()
        test_access_count_increments()
        test_access_persists_across_instances()
        test_memory_types_filter()
        test_over_fetch()
        test_mixed_store_retrieval()
        test_query_recent_and_time_range_return_episodic_only()
        test_temporal_queries_update_episodic_access_counts()
        print("\nAll tests passed.")
    finally:
        cleanup()
