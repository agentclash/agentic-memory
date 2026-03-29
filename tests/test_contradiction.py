"""Verify contradiction candidate retrieval and supersession persistence."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from events.bus import EventBus
from forgetting.contradiction import ContradictionDetector
from models.semantic import SemanticMemory
from stores.media_store import MediaStore
from stores.semantic_store import SemanticStore


class FixedVectorEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_text(self, text: str) -> list[float]:
        return list(self._vectors[text])

    def embed_query(self, text: str) -> list[float]:
        return list(self._vectors[text])


class EventRecorder:
    def __init__(self, bus: EventBus, *event_types: str):
        self.events = []
        for event_type in event_types:
            bus.subscribe(event_type, self.events.append)


def fresh_setup(*, embedder, event_bus: EventBus | None = None):
    db_path = tempfile.mkdtemp(prefix="chroma_test_contradiction_")
    media_root = tempfile.mkdtemp(prefix="semantic_media_")
    shutil.rmtree(db_path, ignore_errors=True)
    original_db_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = db_path
    store = SemanticStore(
        event_bus=event_bus,
        embedder=embedder,
        media_store=MediaStore(media_root),
    )
    return store, db_path, media_root, original_db_path


def cleanup(db_path: str, media_root: str, original_db_path: str) -> None:
    config.CHROMA_DB_PATH = original_db_path
    shutil.rmtree(db_path, ignore_errors=True)
    shutil.rmtree(media_root, ignore_errors=True)


def test_find_potential_contradictions_returns_related_candidates_only():
    vectors = {
        "The feature flag is enabled in production": [1.0, 0.0, 0.0],
        "The feature flag is disabled in production": [0.92, 0.08, 0.0],
        "Paris is the capital of France": [0.0, 1.0, 0.0],
    }
    store, db_path, media_root, original_db_path = fresh_setup(embedder=FixedVectorEmbedder(vectors))
    detector = ContradictionDetector(store)

    try:
        new_record = SemanticMemory(content="The feature flag is enabled in production")
        old_conflicting = SemanticMemory(content="The feature flag is disabled in production")
        unrelated = SemanticMemory(content="Paris is the capital of France")
        store.store(old_conflicting)
        store.store(unrelated)
        store.store(new_record)

        candidates = detector.find_potential_contradictions(new_record, threshold=0.85, top_k=5)

        assert len(candidates) == 1
        assert candidates[0].record.id == old_conflicting.id
        assert candidates[0].similarity > 0.85
        print("  PASS  contradiction lookup returns related candidates but excludes unrelated facts")
    finally:
        cleanup(db_path, media_root, original_db_path)


def test_find_potential_contradictions_emits_flag_event():
    vectors = {
        "Mercury is the closest planet to the Sun": [1.0, 0.0, 0.0],
        "Venus is the closest planet to the Sun": [0.9, 0.1, 0.0],
    }
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.contradiction_flagged")
    store, db_path, media_root, original_db_path = fresh_setup(
        embedder=FixedVectorEmbedder(vectors),
        event_bus=bus,
    )
    detector = ContradictionDetector(store, event_bus=bus)

    try:
        kept = SemanticMemory(content="Mercury is the closest planet to the Sun")
        candidate = SemanticMemory(content="Venus is the closest planet to the Sun")
        store.store(candidate)
        store.store(kept)

        detector.find_potential_contradictions(kept, threshold=0.85, top_k=5)

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert event.data["record_id"] == kept.id
        assert event.data["candidate_ids"] == (candidate.id,)
        assert event.data["candidate_count"] == 1
        print("  PASS  contradiction lookup emits memory.contradiction_flagged when candidates exist")
    finally:
        cleanup(db_path, media_root, original_db_path)


def test_resolve_supersession_zeroes_old_record_and_persists_bidirectional_link():
    vectors = {
        "The service runs in us-east-1": [1.0, 0.0, 0.0],
        "The service runs in eu-west-1": [0.9, 0.1, 0.0],
    }
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.supersession_resolved")
    store, db_path, media_root, original_db_path = fresh_setup(
        embedder=FixedVectorEmbedder(vectors),
        event_bus=bus,
    )
    detector = ContradictionDetector(store, event_bus=bus)

    try:
        old_record = SemanticMemory(content="The service runs in us-east-1", importance=0.8)
        new_record = SemanticMemory(content="The service runs in eu-west-1", importance=0.7)
        store.store(old_record)
        store.store(new_record)

        detector.resolve_supersession(old_record.id, new_record.id)

        reloaded_old = store.get_by_id(old_record.id)
        reloaded_new = store.get_by_id(new_record.id)
        assert reloaded_old is not None
        assert reloaded_new is not None
        assert reloaded_old.importance == 0.0
        assert reloaded_old.superseded_by == new_record.id
        assert reloaded_new.supersedes == old_record.id
        assert len(recorder.events) == 1
        assert recorder.events[0].data["superseded_id"] == old_record.id
        assert recorder.events[0].data["kept_id"] == new_record.id
        print("  PASS  supersession resolution persists pruning state through store.replace")
    finally:
        cleanup(db_path, media_root, original_db_path)


def test_resolve_supersession_overwrites_kept_pointer_with_latest_predecessor():
    vectors = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.9, 0.1, 0.0],
        "C": [0.8, 0.2, 0.0],
    }
    store, db_path, media_root, original_db_path = fresh_setup(embedder=FixedVectorEmbedder(vectors))
    detector = ContradictionDetector(store)

    try:
        first = SemanticMemory(content="A")
        second = SemanticMemory(content="B")
        third = SemanticMemory(content="C")
        store.store(first)
        store.store(second)
        store.store(third)

        detector.resolve_supersession(first.id, second.id)
        detector.resolve_supersession(second.id, third.id)

        reloaded_second = store.get_by_id(second.id)
        reloaded_third = store.get_by_id(third.id)
        assert reloaded_second is not None
        assert reloaded_third is not None
        assert reloaded_second.superseded_by == third.id
        assert reloaded_third.supersedes == second.id
        print("  PASS  supersession keeps only the latest predecessor link on the kept record")
    finally:
        cleanup(db_path, media_root, original_db_path)


def test_find_likely_duplicates_batch_returns_only_very_high_similarity_pairs():
    vectors = {
        "Kubernetes runs containers": [1.0, 0.0, 0.0],
        "Kubernetes runs containers in clusters": [0.99, 0.01, 0.0],
        "Redis is an in-memory data store": [0.0, 0.0, 1.0],
        "The moon orbits Earth": [0.0, 1.0, 0.0],
    }
    store, db_path, media_root, original_db_path = fresh_setup(embedder=FixedVectorEmbedder(vectors))
    detector = ContradictionDetector(store)

    try:
        first = SemanticMemory(content="Kubernetes runs containers")
        second = SemanticMemory(content="Kubernetes runs containers in clusters")
        third = SemanticMemory(content="Redis is an in-memory data store")
        fourth = SemanticMemory(content="The moon orbits Earth")
        for record in (first, second, third, fourth):
            store.store(record)

        duplicates = detector.find_likely_duplicates_batch(threshold=0.95)

        assert len(duplicates) == 1
        assert {duplicates[0][0], duplicates[0][1]} == {first.id, second.id}
        assert duplicates[0][2] > 0.95
        print("  PASS  batch duplicate detection returns only very-high-similarity pairs")
    finally:
        cleanup(db_path, media_root, original_db_path)


if __name__ == "__main__":
    print("Contradiction detector tests:\n")
    test_find_potential_contradictions_returns_related_candidates_only()
    test_find_potential_contradictions_emits_flag_event()
    test_resolve_supersession_zeroes_old_record_and_persists_bidirectional_link()
    test_resolve_supersession_overwrites_kept_pointer_with_latest_predecessor()
    test_find_likely_duplicates_batch_returns_only_very_high_similarity_pairs()
    print("\nAll tests passed.")
