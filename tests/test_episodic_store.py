"""Verify EpisodicMemory defaults and EpisodicStore persistence offline."""

import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from events.bus import EventBus
from models.episodic import EpisodicMemory
from stores.episodic_store import EpisodicStore, EpisodicStoreError, MediaTooLargeError
from tests.helpers import HashingEmbedder

_TEMP_DIRS = []


class EventRecorder:
    def __init__(self, bus: EventBus, *event_types: str):
        self.events = []
        for event_type in event_types:
            bus.subscribe(event_type, self.events.append)


class FailingMediaEmbedder(HashingEmbedder):
    def embed_video(self, video_bytes: bytes, mime_type: str = "video/mp4") -> list[float]:
        raise RuntimeError("provider rejected media payload")


def fresh_setup(*, event_bus: EventBus | None = None, embedder=None, max_media_bytes: int | None = None):
    db_path = tempfile.mkdtemp(prefix="chroma_test_episodic_")
    _TEMP_DIRS.append(db_path)
    shutil.rmtree(db_path, ignore_errors=True)
    config.CHROMA_DB_PATH = db_path
    store = EpisodicStore(
        event_bus=event_bus,
        embedder=embedder or HashingEmbedder(),
        max_media_bytes=max_media_bytes,
    )
    return store, db_path


def make_media_file(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="episodic_media_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(data)
    _TEMP_DIRS.append(path)
    return path


def test_model_defaults():
    record = EpisodicMemory(content="Discussed retrieval metrics")

    assert record.memory_type == "episodic"
    assert record.modality == "text"
    assert str(uuid.UUID(record.session_id)) == record.session_id
    assert record.turn_number is None
    assert record.participants == ["user", "agent"]
    print("  PASS  EpisodicMemory defaults to episodic with sane field defaults")


def test_text_episode_round_trip():
    store, _ = fresh_setup()
    created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    record = EpisodicMemory(
        content="Resolved a ranking bug in the retriever",
        session_id="session-alpha",
        turn_number=7,
        participants=["atharva", "codex"],
        summary="Fixed a scoring regression",
        emotional_valence=0.6,
        created_at=created_at,
        importance=0.9,
    )

    record_id = store.store(record)
    loaded = store.get_by_id(record_id)

    assert loaded is not None
    assert loaded.id == record.id
    assert loaded.session_id == "session-alpha"
    assert loaded.turn_number == 7
    assert loaded.participants == ["atharva", "codex"]
    assert loaded.summary == "Fixed a scoring regression"
    assert loaded.emotional_valence == 0.6
    assert loaded.created_at == created_at
    assert loaded.importance == 0.9
    assert loaded.modality == "text"
    print("  PASS  text-backed episodic records round-trip through the store")


def test_store_emits_memory_stored_on_success():
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.stored")
    store, _ = fresh_setup(event_bus=bus)
    record = EpisodicMemory(
        content="Captured an episodic event after a successful run",
        session_id="session-success",
        importance=0.8,
    )

    record_id = store.store(record)

    assert record_id == record.id
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.data["record_id"] == record.id
    assert event.data["memory_type"] == "episodic"
    assert event.data["content"] == record.content
    assert event.data["modality"] == "text"
    assert event.data["importance"] == 0.8
    assert event.data["session_id"] == "session-success"
    print("  PASS  EpisodicStore emits memory.stored after successful persistence")


def test_media_backed_episode_round_trip():
    store, _ = fresh_setup()
    media_path = make_media_file(".png", b"not-a-real-png-but-good-enough-for-tests")
    record = EpisodicMemory(
        content="Screenshot of the failing deployment",
        session_id="session-media",
        modality="image",
        media_ref=media_path,
        source_mime_type="image/png",
        text_description="CI output showing a dependency resolution error",
    )

    record_id = store.store(record)
    loaded = store.get_by_id(record_id)

    assert loaded is not None
    assert loaded.memory_type == "episodic"
    assert loaded.session_id == "session-media"
    assert loaded.modality == "image"
    assert loaded.media_ref == media_path
    assert loaded.source_mime_type == "image/png"
    assert loaded.text_description == "CI output showing a dependency resolution error"
    print("  PASS  media-backed episodic records preserve multimodal metadata")


def test_get_by_session_orders_by_turn_number_then_created_at():
    store, _ = fresh_setup()
    store.store(
        EpisodicMemory(
            content="turn one, created later",
            session_id="session-alpha",
            turn_number=1,
            created_at=datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="turn two, created earliest",
            session_id="session-alpha",
            turn_number=2,
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="turn one, created earliest",
            session_id="session-alpha",
            turn_number=1,
            created_at=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
        )
    )

    results = store.get_by_session("session-alpha")

    assert [record.content for record in results] == [
        "turn one, created earliest",
        "turn one, created later",
        "turn two, created earliest",
    ]
    print("  PASS  session queries sort by turn_number then created_at")


def test_get_by_session_filters_across_sessions():
    store, _ = fresh_setup()
    store.store(
        EpisodicMemory(
            content="alpha event",
            session_id="session-alpha",
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="beta event",
            session_id="session-beta",
            created_at=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="alpha follow-up",
            session_id="session-alpha",
            created_at=datetime(2026, 1, 1, 12, 2, tzinfo=timezone.utc),
        )
    )

    results = store.get_by_session("session-alpha")

    assert [record.content for record in results] == ["alpha event", "alpha follow-up"]
    print("  PASS  session queries return only matching episodes")


def test_get_recent_returns_newest_first_across_sessions():
    store, _ = fresh_setup()
    media_path = make_media_file(".png", b"recent-image")
    store.store(
        EpisodicMemory(
            content="oldest",
            session_id="session-one",
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="middle media-backed",
            session_id="session-two",
            modality="image",
            media_ref=media_path,
            source_mime_type="image/png",
            created_at=datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc),
        )
    )
    store.store(
        EpisodicMemory(
            content="newest",
            session_id="session-three",
            created_at=datetime(2026, 1, 1, 12, 20, tzinfo=timezone.utc),
        )
    )

    results = store.get_recent(2)

    assert [record.content for record in results] == ["newest", "middle media-backed"]
    print("  PASS  recent queries return newest-first across sessions")


def test_get_by_time_range_is_inclusive_and_treats_naive_datetimes_as_utc():
    store, _ = fresh_setup()
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    store.store(EpisodicMemory(content="before", session_id="session-time", created_at=base - timedelta(minutes=5)))
    store.store(EpisodicMemory(content="start", session_id="session-time", created_at=base))
    store.store(
        EpisodicMemory(
            content="middle",
            session_id="session-time",
            created_at=base + timedelta(minutes=10),
        )
    )
    store.store(EpisodicMemory(content="end", session_id="session-time", created_at=base + timedelta(minutes=15)))
    store.store(
        EpisodicMemory(
            content="after",
            session_id="session-time",
            created_at=base + timedelta(minutes=20),
        )
    )

    results = store.get_by_time_range(
        datetime(2026, 1, 1, 12, 0),
        datetime(2026, 1, 1, 12, 15, tzinfo=timezone.utc),
    )

    assert [record.content for record in results] == ["start", "middle", "end"]
    print("  PASS  time-range queries are inclusive and normalize naive datetimes as UTC")


def test_mixed_modality_session_query_returns_text_and_media_records():
    store, _ = fresh_setup()
    media_path = make_media_file(".png", b"mixed-modality")
    store.store(
        EpisodicMemory(
            content="text event",
            session_id="session-mixed",
            turn_number=1,
        )
    )
    store.store(
        EpisodicMemory(
            content="image event",
            session_id="session-mixed",
            turn_number=2,
            modality="image",
            media_ref=media_path,
            source_mime_type="image/png",
        )
    )

    results = store.get_by_session("session-mixed")

    assert [record.modality for record in results] == ["text", "image"]
    assert [record.content for record in results] == ["text event", "image event"]
    print("  PASS  session queries behave the same for text and media-backed episodes")


def test_retrieval_shape():
    store, _ = fresh_setup()
    store.store(EpisodicMemory(content="Reviewed the deployment checklist"))
    store.store(EpisodicMemory(content="Debugged the memory retrieval stack"))

    results = store.retrieve("memory retrieval", top_k=2)

    assert len(results) == 2
    record, similarity = results[0]
    assert isinstance(record, EpisodicMemory)
    assert isinstance(similarity, float)
    assert record.memory_type == "episodic"
    print("  PASS  retrieve() returns episodic (record, similarity) pairs")


def test_access_tracking():
    store, _ = fresh_setup()
    record_id = store.store(EpisodicMemory(content="Traced an access counter regression"))

    before = store.get_by_id(record_id)
    assert before.access_count == 0
    assert before.last_accessed_at is None

    store.update_access(record_id)

    after = store.get_by_id(record_id)
    assert after.access_count == 1
    assert after.last_accessed_at is not None
    print("  PASS  update_access() increments count and stamps last_accessed_at")


def test_empty_store():
    store, _ = fresh_setup()

    assert store.get_by_id("missing") is None
    assert store.retrieve("nothing here", top_k=3) == []
    assert store.get_by_session("missing-session") == []
    assert store.get_recent(3) == []
    assert store.get_by_time_range(
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    ) == []
    store.update_access("missing")
    print("  PASS  empty store queries return clean empty results across direct query APIs")


def test_oversized_media_fails_before_read_and_does_not_emit_event():
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.stored")
    store, _ = fresh_setup(event_bus=bus, max_media_bytes=8)
    media_path = make_media_file(".mp4", b"0123456789")

    try:
        store.store(
            EpisodicMemory(
                content="Long video of the debugging session",
                session_id="session-video",
                modality="video",
                media_ref=media_path,
                source_mime_type="video/mp4",
            )
        )
        raise AssertionError("Expected oversized media to be rejected")
    except MediaTooLargeError as exc:
        message = str(exc)
        assert "size_bytes=10" in message
        assert "limit_bytes=8" in message

    assert recorder.events == []
    assert store.retrieve("debugging", top_k=5) == []
    print("  PASS  oversized media is rejected cleanly before persistence")


def test_media_provider_errors_are_wrapped_and_do_not_emit_event():
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.stored")
    store, _ = fresh_setup(event_bus=bus, embedder=FailingMediaEmbedder())
    media_path = make_media_file(".mp4", b"small-video")

    record = EpisodicMemory(
        content="Short video of a failed run",
        session_id="session-video",
        modality="video",
        media_ref=media_path,
        source_mime_type="video/mp4",
    )

    record_id = store.store(record)
    loaded = store.get_by_id(record_id)

    assert loaded is not None
    assert loaded.metadata["embedding_strategy"] == "text_fallback"
    assert "provider rejected media payload" in loaded.metadata["media_embed_error"]
    assert len(recorder.events) == 1
    print("  PASS  provider failures fall back to text embedding and still persist the record")


def cleanup():
    for path in _TEMP_DIRS:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    print("Episodic store tests:\n")
    try:
        test_model_defaults()
        test_text_episode_round_trip()
        test_store_emits_memory_stored_on_success()
        test_media_backed_episode_round_trip()
        test_get_by_session_orders_by_turn_number_then_created_at()
        test_get_by_session_filters_across_sessions()
        test_get_recent_returns_newest_first_across_sessions()
        test_get_by_time_range_is_inclusive_and_treats_naive_datetimes_as_utc()
        test_mixed_modality_session_query_returns_text_and_media_records()
        test_retrieval_shape()
        test_access_tracking()
        test_empty_store()
        test_oversized_media_fails_before_read_and_does_not_emit_event()
        test_media_provider_errors_are_wrapped_and_do_not_emit_event()
        print("\nAll tests passed.")
    finally:
        cleanup()
