"""Verify SemanticMemory persistence, including multimodal media writes."""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.semantic import SemanticMemory
from stores.media_store import MediaStore
from stores.semantic_store import SemanticStore
from tests.helpers import DeterministicMultimodalEmbedder, HashingEmbedder

_TEMP_DIRS = []


class RecordingSemanticEmbedder(HashingEmbedder):
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
            (
                "multimodal",
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
                },
            )
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


def fresh_setup(*, embedder=None):
    db_path = tempfile.mkdtemp(prefix="chroma_test_semantic_")
    media_root = tempfile.mkdtemp(prefix="semantic_media_")
    _TEMP_DIRS.extend([db_path, media_root])
    shutil.rmtree(db_path, ignore_errors=True)
    config.CHROMA_DB_PATH = db_path
    store = SemanticStore(
        embedder=embedder or HashingEmbedder(),
        media_store=MediaStore(media_root),
    )
    return store, media_root


def make_media_file(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="semantic_media_source_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(data)
    _TEMP_DIRS.append(path)
    return path


def test_text_semantic_round_trip_preserves_metadata():
    store, _ = fresh_setup()
    created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    record = SemanticMemory(
        content="HTTP uses request-response semantics",
        created_at=created_at,
        category="networking",
        domain="web",
        confidence=0.8,
        supersedes="old-fact",
        related_ids=["fact-1", "fact-2"],
        has_visual=True,
    )

    record_id = store.store(record)
    loaded = store.get_by_id(record_id)

    assert loaded is not None
    assert loaded.category == "networking"
    assert loaded.domain == "web"
    assert loaded.confidence == 0.8
    assert loaded.supersedes == "old-fact"
    assert loaded.related_ids == ["fact-1", "fact-2"]
    assert loaded.has_visual is True
    assert loaded.created_at == created_at
    print("  PASS  text semantic records round-trip full semantic metadata")


def test_image_semantic_write_copies_media_and_uses_owned_path():
    embedder = RecordingSemanticEmbedder()
    store, media_root = fresh_setup(embedder=embedder)
    source_path = make_media_file(".png", b"semantic-image")
    record = SemanticMemory(
        content="Architecture diagram for the retrieval stack",
        modality="image",
        media_ref=source_path,
        media_type="image",
        text_description="Whiteboard with retriever, ranker, and stores",
    )

    store.store(record)
    loaded = store.get_by_id(record.id)

    assert loaded is not None
    assert loaded.media_ref == os.path.join(media_root, "images", f"{record.id}.png")
    assert os.path.exists(loaded.media_ref)
    assert Path(loaded.media_ref).read_bytes() == b"semantic-image"
    assert embedder.calls == [
        (
            "multimodal",
            {
                "text": "Architecture diagram for the retrieval stack\nWhiteboard with retriever, ranker, and stores",
                "image": loaded.media_ref,
                "audio": None,
                "video": None,
                "pdf": None,
                "image_mime_type": "image/png",
                "audio_mime_type": "audio/mpeg",
                "video_mime_type": "video/mp4",
                "pdf_mime_type": "application/pdf",
            },
        )
    ]
    print("  PASS  semantic image writes copy media and embed through the multimodal path")


def test_multimodal_semantic_write_stores_one_vector_and_round_trips():
    embedder = RecordingSemanticEmbedder()
    store, media_root = fresh_setup(embedder=embedder)
    source_path = make_media_file(".pdf", b"%PDF-1.4\nsemantic")
    record = SemanticMemory(
        content="Incident review handoff",
        modality="multimodal",
        media_ref=source_path,
        media_type="pdf",
        text_description="Timeline and remediation notes",
        related_ids=["incident-1"],
    )

    store.store(record)
    loaded = store.get_by_id(record.id)
    raw = store._collection.get(ids=[record.id], include=["embeddings", "metadatas"])

    assert loaded is not None
    assert loaded.modality == "multimodal"
    assert loaded.media_ref == os.path.join(media_root, "documents", f"{record.id}.pdf")
    assert loaded.media_type == "pdf"
    assert loaded.related_ids == ["incident-1"]
    assert len(raw["embeddings"]) == 1
    assert embedder.calls == [
        (
            "multimodal",
            {
                "text": "Incident review handoff\nTimeline and remediation notes",
                "image": None,
                "audio": None,
                "video": None,
                "pdf": loaded.media_ref,
                "image_mime_type": "image/png",
                "audio_mime_type": "audio/mpeg",
                "video_mime_type": "video/mp4",
                "pdf_mime_type": "application/pdf",
            },
        )
    ]
    print("  PASS  multimodal semantic writes store one vector and round-trip metadata")


def test_semantic_store_cleans_up_owned_media_on_failure():
    embedder = RecordingSemanticEmbedder()
    store, media_root = fresh_setup(embedder=embedder)
    source_path = make_media_file(".png", b"semantic-image")
    record = SemanticMemory(
        content="This semantic write should fail",
        modality="image",
        media_ref=source_path,
        media_type="image",
    )

    def fail_add(*args, **kwargs):
        raise RuntimeError("boom")

    store._collection.add = fail_add

    try:
        store.store(record)
        raise AssertionError("Expected store() to raise")
    except RuntimeError:
        pass

    assert not any(path.is_file() for path in Path(media_root).rglob("*"))
    print("  PASS  failed semantic writes clean up owned media")


def test_get_all_records_returns_full_collection_without_embeddings_by_default():
    store, _ = fresh_setup()
    first = SemanticMemory(content="First fact", importance=0.2)
    second = SemanticMemory(content="Second fact", importance=0.7)
    store.store(first)
    store.store(second)

    records = store.get_all_records()

    assert {record.id for record in records} == {first.id, second.id}
    assert all(record.embedding is None for record in records)
    print("  PASS  semantic scan returns every record and omits embeddings by default")


def test_semantic_delete_is_idempotent_and_replace_preserves_embedding():
    embedder = RecordingSemanticEmbedder()
    store, _ = fresh_setup(embedder=embedder)
    record = SemanticMemory(content="Original fact", importance=0.9, category="ops")
    store.store(record)
    original_embedding = list(record.embedding)

    record.content = "Updated fact"
    record.importance = 0.3
    record.supersedes = "old-fact"
    store.replace(record)
    loaded = store.get_by_id(record.id)
    store.delete("missing-id")

    assert embedder.calls == []
    assert loaded is not None
    assert loaded.content == "Updated fact"
    assert loaded.importance == 0.3
    assert loaded.supersedes == "old-fact"
    assert loaded.embedding == original_embedding

    store.delete(record.id)
    assert store.get_by_id(record.id) is None
    print("  PASS  semantic replace rewrites metadata without re-embedding and delete is idempotent")


def test_retrieve_by_vector_supports_image_embedding_against_text_memory():
    embedder = DeterministicMultimodalEmbedder()
    store, _ = fresh_setup(embedder=embedder)
    image_path = make_media_file(".png", b"diagram")
    record = SemanticMemory(content="image png diagram architecture memory")

    store.store(record)
    query_vector = embedder.embed_image(image_path, mime_type="image/png")
    results = store.retrieve_by_vector(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0][0].id == record.id
    assert isinstance(results[0][1], float)
    print("  PASS  semantic vector retrieval can match an image embedding to a text memory")


if __name__ == "__main__":
    print("Semantic store tests:\n")
    test_text_semantic_round_trip_preserves_metadata()
    test_image_semantic_write_copies_media_and_uses_owned_path()
    test_multimodal_semantic_write_stores_one_vector_and_round_trips()
    test_semantic_store_cleans_up_owned_media_on_failure()
    test_get_all_records_returns_full_collection_without_embeddings_by_default()
    test_semantic_delete_is_idempotent_and_replace_preserves_embedding()
    test_retrieve_by_vector_supports_image_embedding_against_text_memory()
    print("\nAll tests passed.")
