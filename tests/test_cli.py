"""Verify CLI episodic commands without provider-backed dependencies."""

import io
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import cli
from events.bus import EventBus
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from stores.media_store import MediaStore


class RecordingStore:
    def __init__(self):
        self.records = []

    def store(self, record):
        self.records.append(record)
        return record.id


class FailingStore:
    def store(self, record):
        raise RuntimeError("synthetic store failure")


class FakeRetriever:
    def __init__(self, recent_records=None, vector_results=None):
        self.recent_records = recent_records or []
        self.vector_results = vector_results or []
        self.recent_calls = []
        self.vector_calls = []

    def query_recent(self, n: int):
        self.recent_calls.append(n)
        return self.recent_records[:n]

    def query_by_vector(self, vector, top_k=5, memory_types=None, metadata=None):
        self.vector_calls.append(
            {
                "vector": vector,
                "top_k": top_k,
                "memory_types": memory_types,
                "metadata": metadata,
            }
        )
        return self.vector_results[:top_k]


class FakeEmbedder:
    def __init__(self):
        self.image_calls = []
        self.audio_calls = []

    def embed_image(self, source, mime_type=None):
        self.image_calls.append((source, mime_type))
        return [0.25, 0.75]

    def embed_audio(self, source, mime_type=None):
        self.audio_calls.append((source, mime_type))
        return [0.6, 0.4]


def run_cli(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch.object(sys, "argv", argv):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            cli.main()
    return stdout.getvalue(), stderr.getvalue()


def test_store_episode_text_cli_smoke():
    store = RecordingStore()

    with patch.object(cli, "_make_bus", return_value=EventBus()):
        with patch.object(cli, "_make_episodic_store", return_value=store):
            stdout, _ = run_cli(
                [
                    "cli",
                    "store-episode",
                    "--session",
                    "session-text",
                    "--text",
                    "We resolved an episodic retrieval bug",
                ]
            )

    assert "Stored episode [" in stdout
    assert len(store.records) == 1
    record = store.records[0]
    assert isinstance(record, EpisodicMemory)
    assert record.content == "We resolved an episodic retrieval bug"
    assert record.session_id == "session-text"
    assert record.modality == "text"
    print("  PASS  CLI stores text-backed episodic memories")


def test_store_episode_file_cli_smoke():
    store = RecordingStore()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cli_episode_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"fake-image")

    try:
        with tempfile.TemporaryDirectory(prefix="cli_media_root_") as media_root:
            with patch.object(cli, "_make_bus", return_value=EventBus()):
                with patch.object(cli, "_make_episodic_store", return_value=store):
                    with patch.object(cli, "_make_media_store", return_value=MediaStore(media_root)):
                        stdout, _ = run_cli(
                            [
                                "cli",
                                "store-episode",
                                "--session",
                                "session-file",
                                "--file",
                                path,
                                "--modality",
                                "image",
                                "--content",
                                "Screenshot from the failed run",
                            ]
                        )

            assert "Stored episode [" in stdout
            assert len(store.records) == 1
            record = store.records[0]
            assert record.content == "Screenshot from the failed run"
            assert record.session_id == "session-file"
            assert record.modality == "image"
            assert record.media_ref == str(Path(media_root) / "images" / f"{record.id}.png")
            assert Path(record.media_ref).read_bytes() == b"fake-image"
            assert record.source_mime_type == "image/png"
            print("  PASS  CLI copies file-backed episodic media into app-owned storage")
    finally:
        os.remove(path)


def test_store_semantic_image_cli_uses_required_content_and_owned_media():
    store = RecordingStore()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cli_semantic_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"semantic-image")

    try:
        with tempfile.TemporaryDirectory(prefix="cli_media_root_") as media_root:
            with patch.object(cli, "_make_bus", return_value=EventBus()):
                with patch.object(cli, "_make_semantic_store", return_value=store):
                    stdout, _ = run_cli(
                        [
                            "cli",
                            "store",
                            "Architecture whiteboard",
                            "--image",
                            path,
                        ]
                    )

            assert "Stored [" in stdout
            assert len(store.records) == 1
            record = store.records[0]
            assert isinstance(record, SemanticMemory)
            assert record.content == "Architecture whiteboard"
            assert record.modality == "image"
            assert record.media_ref == os.path.abspath(path)
            assert record.media_type == "image"
            print("  PASS  CLI semantic store keeps content required for image-backed facts")
    finally:
        os.remove(path)


def test_recent_cli_smoke():
    record = EpisodicMemory(
        content="Most recent episode",
        session_id="session-recent",
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    retriever = FakeRetriever(recent_records=[record])

    with patch.object(cli, "_make_bus", return_value=EventBus()):
        with patch.object(cli, "_make_retriever", return_value=retriever):
            stdout, _ = run_cli(["cli", "recent", "1"])

    assert retriever.recent_calls == [1]
    assert "Most recent episode" in stdout
    assert "session=session-recent" in stdout
    print("  PASS  CLI prints recent episodic memories")


def test_query_by_image_cli_embeds_file_and_prints_media_context():
    record = SemanticMemory(
        content="Architecture diagram memory",
        modality="image",
        media_ref="/tmp/owned-diagram.png",
        media_type="image",
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    result = type(
        "RankedResult",
        (),
        {
            "record": record,
            "final_score": 0.92,
            "raw_similarity": 0.88,
            "recency_score": 0.40,
            "importance_score": 0.50,
        },
    )()
    retriever = FakeRetriever(vector_results=[result])
    embedder = FakeEmbedder()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cli_query_image_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"query-image")

    try:
        with patch.object(cli, "_make_bus", return_value=EventBus()):
            with patch.object(cli, "_make_retriever", return_value=retriever):
                with patch.object(cli, "_make_embedder", return_value=embedder):
                    stdout, _ = run_cli(["cli", "query-by-image", path, "--memory-types", "semantic"])

        assert embedder.image_calls == [(os.path.abspath(path), "image/png")]
        assert retriever.vector_calls == [
            {
                "vector": [0.25, 0.75],
                "top_k": 5,
                "memory_types": ["semantic"],
                "metadata": {"source_modality": "image"},
            }
        ]
        assert "Architecture diagram memory" in stdout
        assert "modality=image" in stdout
        assert "media=/tmp/owned-diagram.png" in stdout
        print("  PASS  CLI query-by-image uses vector retrieval and prints media context")
    finally:
        os.remove(path)


def test_store_episode_multimodal_cli_infers_media_type_from_file():
    store = RecordingStore()
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="cli_episode_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"%PDF-1.4\nfake")

    try:
        with tempfile.TemporaryDirectory(prefix="cli_media_root_") as media_root:
            with patch.object(cli, "_make_bus", return_value=EventBus()):
                with patch.object(cli, "_make_episodic_store", return_value=store):
                    with patch.object(cli, "_make_media_store", return_value=MediaStore(media_root)):
                        stdout, _ = run_cli(
                            [
                                "cli",
                                "store-episode",
                                "--session",
                                "session-multimodal",
                                "--file",
                                path,
                                "--modality",
                                "multimodal",
                            ]
                        )

            assert "Stored episode [" in stdout
            assert len(store.records) == 1
            record = store.records[0]
            assert record.modality == "multimodal"
            assert record.media_type == "pdf"
            assert record.source_mime_type == "application/pdf"
            print("  PASS  CLI multimodal episodic writes infer media_type from the file")
    finally:
        os.remove(path)


def test_store_episode_file_cli_cleans_up_owned_media_on_failure():
    store = FailingStore()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cli_episode_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"fake-image")

    try:
        with tempfile.TemporaryDirectory(prefix="cli_media_root_") as media_root:
            with patch.object(cli, "_make_bus", return_value=EventBus()):
                with patch.object(cli, "_make_episodic_store", return_value=store):
                    with patch.object(cli, "_make_media_store", return_value=MediaStore(media_root)):
                        try:
                            run_cli(
                                [
                                    "cli",
                                    "store-episode",
                                    "--session",
                                    "session-file",
                                    "--file",
                                    path,
                                    "--modality",
                                    "image",
                                ]
                            )
                            raise AssertionError("Expected CLI store failure")
                        except RuntimeError as exc:
                            assert "synthetic store failure" in str(exc)

            assert not any(file_path.is_file() for file_path in Path(media_root).rglob("*"))
            print("  PASS  CLI cleans up owned media when store persistence fails")
    finally:
        os.remove(path)


if __name__ == "__main__":
    print("CLI tests:\n")
    test_store_episode_text_cli_smoke()
    test_store_episode_file_cli_smoke()
    test_store_semantic_image_cli_uses_required_content_and_owned_media()
    test_recent_cli_smoke()
    test_query_by_image_cli_embeds_file_and_prints_media_context()
    test_store_episode_multimodal_cli_infers_media_type_from_file()
    test_store_episode_file_cli_cleans_up_owned_media_on_failure()
    print("\nAll tests passed.")
