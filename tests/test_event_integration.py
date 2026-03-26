"""Verify store and retriever event emissions without external APIs."""

import os
import sys
from dataclasses import replace
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from events.bus import EventBus
from models.semantic import SemanticMemory
from retrieval.retriever import UnifiedRetriever
from stores.base import BaseStore
from stores.semantic_store import SemanticStore
from tests.helpers import HashingEmbedder, cleanup_dir, make_temp_chroma_dir


class EventRecorder:
    def __init__(self, bus: EventBus, *event_types: str):
        self.events = []
        for event_type in event_types:
            bus.subscribe(event_type, self.events.append)


class FakeStore(BaseStore):
    def __init__(self, memory_type: str, results: list[tuple[SemanticMemory, float]]):
        super().__init__()
        self._memory_type = memory_type
        self._results = results
        self.updated_ids = []

    def store(self, record):
        self._results.append((record, 1.0))
        return record.id

    def get_by_id(self, record_id):
        for record, _ in self._results:
            if record.id == record_id:
                return record
        return None

    def retrieve(self, query: str, top_k: int = 5):
        return self._results[:top_k]

    def update_access(self, record_id: str) -> None:
        self.updated_ids.append(record_id)


def test_semantic_store_emits_memory_stored():
    db_path = make_temp_chroma_dir("chroma_test_event_store_")
    original_db_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = db_path

    try:
        bus = EventBus()
        recorder = EventRecorder(bus, "memory.stored")
        store = SemanticStore(event_bus=bus, embedder=HashingEmbedder())
        record = SemanticMemory(content="Python was created by Guido van Rossum", importance=0.8)

        record_id = store.store(record)

        assert record_id == record.id
        assert len(recorder.events) == 1, f"Expected 1 event, got {len(recorder.events)}"
        event = recorder.events[0]
        assert event.data["record_id"] == record.id
        assert event.data["memory_type"] == "semantic"
        assert event.data["content"] == record.content
        assert event.data["modality"] == "text"
        assert event.data["importance"] == 0.8
        print("  PASS  SemanticStore emits memory.stored after persistence")
    finally:
        config.CHROMA_DB_PATH = original_db_path
        cleanup_dir(db_path)


def test_semantic_store_does_not_emit_when_write_fails():
    db_path = make_temp_chroma_dir("chroma_test_event_store_fail_")
    original_db_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = db_path

    try:
        bus = EventBus()
        recorder = EventRecorder(bus, "memory.stored")
        store = SemanticStore(event_bus=bus, embedder=HashingEmbedder())
        record = SemanticMemory(content="This write should fail")

        def fail_add(*args, **kwargs):
            raise RuntimeError("boom")

        store._collection.add = fail_add

        try:
            store.store(record)
            raise AssertionError("Expected store() to raise")
        except RuntimeError:
            pass

        assert recorder.events == [], "No event should be emitted on failed persistence"
        print("  PASS  failed writes do not emit memory.stored")
    finally:
        config.CHROMA_DB_PATH = original_db_path
        cleanup_dir(db_path)


def test_retriever_emits_retrieved_ranked_and_accessed():
    now = datetime.now(timezone.utc)
    best = SemanticMemory(
        content="Python was created by Guido van Rossum",
        created_at=now - timedelta(minutes=5),
        importance=0.9,
    )
    other = SemanticMemory(
        content="The capital of France is Paris",
        created_at=now - timedelta(days=30),
        importance=0.2,
    )

    semantic_store = FakeStore("semantic", [(best, 0.95), (other, 0.40)])
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.retrieved", "memory.ranked", "memory.accessed")
    retriever = UnifiedRetriever(stores={"semantic": semantic_store}, event_bus=bus)

    results = retriever.query("Who created Python?", top_k=1)

    assert len(results) == 1
    assert [event.event_type for event in recorder.events] == [
        "memory.retrieved",
        "memory.ranked",
        "memory.accessed",
    ]

    retrieved_event, ranked_event, accessed_event = recorder.events
    assert retrieved_event.data["query"] == "Who created Python?"
    assert retrieved_event.data["memory_types"] == ("semantic",)
    assert retrieved_event.data["candidate_count"] == 2
    assert retrieved_event.data["top_similarity"] == 0.95

    assert len(ranked_event.data["results"]) == 1
    assert ranked_event.data["results"][0]["record_id"] == best.id
    assert ranked_event.data["weights"]["relevance"] == 0.4
    assert ranked_event.data["weights"]["recency"] == 0.3
    assert ranked_event.data["weights"]["importance"] == 0.3

    assert accessed_event.data["record_id"] == best.id
    assert accessed_event.data["memory_type"] == "semantic"
    assert accessed_event.data["access_count"] == 1
    assert semantic_store.updated_ids == [best.id]
    print("  PASS  retriever emits retrieved, ranked, and accessed events")


def test_retriever_emits_empty_summary_without_access_events():
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.retrieved", "memory.ranked", "memory.accessed")
    retriever = UnifiedRetriever(stores={"semantic": FakeStore("semantic", [])}, event_bus=bus)

    results = retriever.query("Unknown query", top_k=3)

    assert results == []
    assert [event.event_type for event in recorder.events] == [
        "memory.retrieved",
        "memory.ranked",
    ]
    assert recorder.events[0].data["candidate_count"] == 0
    assert recorder.events[0].data["top_similarity"] is None
    assert recorder.events[1].data["results"] == ()
    print("  PASS  empty retrieval emits summary events but no access events")


def test_retriever_reports_filtered_memory_types():
    now = datetime.now(timezone.utc)
    semantic_record = SemanticMemory(content="Semantic fact", created_at=now, importance=0.5)
    procedural_record = replace(semantic_record, id="procedural-1", memory_type="procedural", content="Procedure")

    semantic_store = FakeStore("semantic", [(semantic_record, 0.8)])
    procedural_store = FakeStore("procedural", [(procedural_record, 0.7)])
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.retrieved")
    retriever = UnifiedRetriever(
        stores={"semantic": semantic_store, "procedural": procedural_store},
        event_bus=bus,
    )

    retriever.query("fact", memory_types=["procedural"], top_k=1)

    assert len(recorder.events) == 1
    assert recorder.events[0].data["memory_types"] == ("procedural",)
    assert recorder.events[0].data["candidate_count"] == 1
    print("  PASS  memory_types filter is reflected in memory.retrieved")


if __name__ == "__main__":
    print("Event integration tests:\n")
    test_semantic_store_emits_memory_stored()
    test_semantic_store_does_not_emit_when_write_fails()
    test_retriever_emits_retrieved_ranked_and_accessed()
    test_retriever_emits_empty_summary_without_access_events()
    test_retriever_reports_filtered_memory_types()
    print("\nAll tests passed.")
