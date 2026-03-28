"""Verify ProceduralMemory persistence, scoring, and multimodal writes offline."""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from events.bus import EventBus
from models.procedural import ProceduralMemory
from stores.media_store import MediaStore
from stores.procedural_store import ProceduralStore
from tests.helpers import HashingEmbedder

_TEMP_DIRS = []


class EventRecorder:
    def __init__(self, bus: EventBus, *event_types: str):
        self.events = []
        for event_type in event_types:
            bus.subscribe(event_type, self.events.append)


class RecordingProceduralEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def embed_multimodal(
        self,
        *,
        text: str | None = None,
        image: str | None = None,
        audio: str | None = None,
        video: str | None = None,
        pdf: str | None = None,
        image_mime_type: str | None = "image/png",
        audio_mime_type: str | None = "audio/mpeg",
        video_mime_type: str | None = "video/mp4",
        pdf_mime_type: str | None = "application/pdf",
    ) -> list[float]:
        self.calls.append(
            {
                "text": text,
                "image": image,
                "audio": audio,
                "video": video,
                "pdf": pdf,
                "image_mime_type": image_mime_type,
                "audio_mime_type": audio_mime_type,
                "video_mime_type": video_mime_type,
                "pdf_mime_type": pdf_mime_type,
            }
        )
        return super().embed_multimodal(
            text=text,
            image=image,
            audio=audio,
            video=video,
            pdf=pdf,
            image_mime_type=image_mime_type,
            audio_mime_type=audio_mime_type,
            video_mime_type=video_mime_type,
            pdf_mime_type=pdf_mime_type,
        )


def fresh_setup(*, embedder=None, event_bus: EventBus | None = None):
    db_path = tempfile.mkdtemp(prefix="chroma_test_procedural_")
    media_root = tempfile.mkdtemp(prefix="procedural_media_")
    _TEMP_DIRS.extend([db_path, media_root])
    shutil.rmtree(db_path, ignore_errors=True)
    config.CHROMA_DB_PATH = db_path
    store = ProceduralStore(
        event_bus=event_bus,
        embedder=embedder or HashingEmbedder(),
        media_store=MediaStore(media_root),
    )
    return store, media_root


def make_media_file(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="procedural_media_source_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(data)
    _TEMP_DIRS.append(path)
    return path


def cleanup() -> None:
    for path in _TEMP_DIRS:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def test_text_procedure_round_trip_preserves_metadata():
    store, _ = fresh_setup()
    created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package dependencies", "Create handler", "Run sam deploy"],
        preconditions=["AWS credentials configured"],
        success_count=5,
        failure_count=2,
        created_at=created_at,
        importance=0.8,
        source="runbook",
        metadata={"owner": "ops"},
    )

    record_id = store.store(record)
    loaded = store.get_by_id(record_id)

    assert loaded is not None
    assert loaded.steps == record.steps
    assert loaded.preconditions == record.preconditions
    assert loaded.success_count == 5
    assert loaded.failure_count == 2
    assert loaded.created_at == created_at
    assert loaded.importance == 0.8
    assert loaded.source == "runbook"
    assert loaded.metadata == {"owner": "ops"}
    print("  PASS  text-backed procedural records round-trip full metadata")


def test_store_emits_memory_stored_with_procedural_fields():
    bus = EventBus()
    recorder = EventRecorder(bus, "memory.stored")
    store, _ = fresh_setup(event_bus=bus)
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Deploy"],
        preconditions=["AWS CLI configured"],
        importance=0.8,
    )

    store.store(record)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.data["memory_type"] == "procedural"
    assert event.data["step_count"] == 2
    assert event.data["precondition_count"] == 1
    assert event.data["importance"] == 0.8
    print("  PASS  ProceduralStore emits standard stored event payload plus procedure counts")


def test_media_backed_procedure_round_trip_uses_combined_embedding_path():
    embedder = RecordingProceduralEmbedder()
    store, media_root = fresh_setup(embedder=embedder)
    source_path = make_media_file(".pdf", b"%PDF-1.4\nrunbook")
    record = ProceduralMemory(
        content="Review migration checklist",
        steps=["Open the checklist", "Validate preconditions", "Run the migration"],
        modality="multimodal",
        media_ref=source_path,
        media_type="pdf",
        text_description="Reference PDF for the migration sequence",
    )

    store.store(record)
    loaded = store.get_by_id(record.id)

    assert loaded is not None
    assert loaded.media_ref == os.path.join(media_root, "documents", f"{record.id}.pdf")
    assert loaded.media_type == "pdf"
    assert loaded.text_description == "Reference PDF for the migration sequence"
    assert embedder.calls == [
        {
            "text": "Review migration checklist\nReference PDF for the migration sequence",
            "image": None,
            "audio": None,
            "video": None,
            "pdf": loaded.media_ref,
            "image_mime_type": "image/png",
            "audio_mime_type": "audio/mpeg",
            "video_mime_type": "video/mp4",
            "pdf_mime_type": "application/pdf",
        }
    ]
    print("  PASS  media-backed procedures use one semantic-style multimodal embedding")


def test_record_outcome_rewrites_metadata_without_reembedding():
    embedder = RecordingProceduralEmbedder()
    store, _ = fresh_setup(embedder=embedder)
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Deploy"],
        success_count=1,
    )
    store.store(record)
    raw_before = store._collection.get(
        ids=[record.id],
        include=["embeddings", "documents", "metadatas"],
    )

    store.record_outcome(record.id, True)

    raw_after = store._collection.get(
        ids=[record.id],
        include=["embeddings", "documents", "metadatas"],
    )
    reloaded = store.get_by_id(record.id)

    assert reloaded is not None
    assert reloaded.success_count == 2
    assert reloaded.failure_count == 0
    assert raw_before["documents"] == raw_after["documents"]
    assert raw_before["embeddings"][0].tolist() == raw_after["embeddings"][0].tolist()
    assert raw_after["metadatas"][0]["success_count"] == 2
    assert raw_after["metadatas"][0]["failure_count"] == 0
    assert embedder.calls == []
    print("  PASS  record_outcome rewrites metadata only and preserves embedding/document data")


def test_record_outcome_missing_record_is_a_no_op():
    store, _ = fresh_setup()

    store.record_outcome("missing-id", True)

    assert store._collection.count() == 0
    print("  PASS  missing procedural ids are ignored during outcome writes")


def test_get_all_records_returns_records_without_embeddings_by_default():
    store, _ = fresh_setup()
    first = ProceduralMemory(content="First procedure", steps=["One"])
    second = ProceduralMemory(content="Second procedure", steps=["Two"])
    store.store(first)
    store.store(second)

    records = store.get_all_records()

    assert {record.id for record in records} == {first.id, second.id}
    assert all(record.embedding is None for record in records)
    print("  PASS  procedural scan returns every record and omits embeddings by default")


def test_procedural_delete_is_idempotent_and_replace_preserves_embedding():
    embedder = RecordingProceduralEmbedder()
    store, _ = fresh_setup(embedder=embedder)
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Deploy"],
        success_count=1,
        importance=0.8,
    )
    store.store(record)
    original_embedding = list(record.embedding)

    record.content = "Deploy to Lambda safely"
    record.importance = 0.4
    record.failure_count = 2
    store.replace(record)
    loaded = store.get_by_id(record.id)
    store.delete("missing-id")
    store.delete(record.id)

    assert embedder.calls == []
    assert loaded is not None
    assert loaded.content == "Deploy to Lambda safely"
    assert loaded.importance == 0.4
    assert loaded.failure_count == 2
    assert loaded.embedding == original_embedding
    assert store.get_by_id(record.id) is None
    print("  PASS  procedural replace rewrites metadata without re-embedding and delete is idempotent")


def test_get_best_procedures_reranks_by_wilson_score():
    store, _ = fresh_setup()
    sam = ProceduralMemory(
        content="Deploy to Lambda via SAM",
        steps=["Package", "Deploy"],
        success_count=9,
        failure_count=1,
    )
    serverless = ProceduralMemory(
        content="Deploy to Lambda via Serverless",
        steps=["Package", "Deploy"],
        success_count=6,
        failure_count=4,
    )
    fresh = ProceduralMemory(
        content="Deploy to Lambda manually",
        steps=["Package", "Deploy"],
    )

    store.retrieve = lambda task, top_k=5: [  # type: ignore[method-assign]
        (sam, 0.9),
        (serverless, 0.9),
        (fresh, 0.4),
    ][:top_k]

    results = store.get_best_procedure_matches("deploy to Lambda", top_k=3)

    assert [result.record.content for result in results] == [
        "Deploy to Lambda via SAM",
        "Deploy to Lambda via Serverless",
        "Deploy to Lambda manually",
    ]
    assert results[0].combined_score > results[1].combined_score > results[2].combined_score
    print("  PASS  best-procedure reranking prefers higher Wilson score when similarity is controlled")


def test_untested_procedures_rank_on_similarity_only():
    store, _ = fresh_setup()
    tested = ProceduralMemory(
        content="Deploy to Lambda via SAM",
        steps=["Package", "Deploy"],
        success_count=3,
        failure_count=1,
    )
    fresh = ProceduralMemory(
        content="Deploy to Lambda manually",
        steps=["Package", "Deploy"],
    )

    store.retrieve = lambda task, top_k=5: [  # type: ignore[method-assign]
        (tested, 0.6),
        (fresh, 0.8),
    ][:top_k]

    results = store.get_best_procedure_matches("deploy to Lambda", top_k=2)

    assert [result.record.content for result in results] == [
        "Deploy to Lambda manually",
        "Deploy to Lambda via SAM",
    ]
    assert results[0].combined_score == 0.8
    assert results[0].wilson_score == 0.0
    print("  PASS  untested procedures compete on similarity alone instead of taking a cold-start penalty")


def test_get_best_procedures_returns_empty_for_empty_store():
    store, _ = fresh_setup()
    assert store.get_best_procedures("deploy", top_k=3) == []
    print("  PASS  empty procedural store returns no best procedures")


if __name__ == "__main__":
    print("Procedural store tests:\n")
    try:
        test_text_procedure_round_trip_preserves_metadata()
        test_store_emits_memory_stored_with_procedural_fields()
        test_media_backed_procedure_round_trip_uses_combined_embedding_path()
        test_record_outcome_rewrites_metadata_without_reembedding()
        test_record_outcome_missing_record_is_a_no_op()
        test_get_all_records_returns_records_without_embeddings_by_default()
        test_procedural_delete_is_idempotent_and_replace_preserves_embedding()
        test_get_best_procedures_reranks_by_wilson_score()
        test_untested_procedures_rank_on_similarity_only()
        test_get_best_procedures_returns_empty_for_empty_store()
        print("\nAll tests passed.")
    finally:
        cleanup()
