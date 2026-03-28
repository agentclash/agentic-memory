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
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from stores.procedural_store import ProceduralMatch
from stores.media_store import MediaStore


class RecordingStore:
    def __init__(self):
        self.records = []

    def store(self, record):
        self.records.append(record)
        return record.id


class RecordingProceduralStore(RecordingStore):
    def __init__(self, record: ProceduralMemory | None = None, matches=None):
        super().__init__()
        self.outcomes = []
        self.record = record
        self.matches = matches or []

    def get_by_id(self, record_id):
        if self.record and self.record.id == record_id:
            return self.record
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def record_outcome(self, record_id, success):
        self.outcomes.append((record_id, success))
        record = self.get_by_id(record_id)
        if record is not None:
            record.record_outcome(success)

    def get_best_procedure_matches(self, task, top_k=3):
        return self.matches[:top_k]


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


def run_cli_expecting_exit(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch.object(sys, "argv", argv):
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                cli.main()
                raise AssertionError("Expected CLI to exit")
            except SystemExit as exc:
                return exc.code, stdout.getvalue(), stderr.getvalue()


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


def test_store_semantic_image_cli_reports_missing_file_cleanly():
    exit_code, stdout, stderr = run_cli_expecting_exit(
        [
            "cli",
            "store",
            "Architecture whiteboard",
            "--image",
            "./does-not-exist.png",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "Media file not found" in stderr
    print("  PASS  CLI semantic store reports missing media without a traceback")


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


def test_store_procedure_cli_smoke():
    store = RecordingProceduralStore()

    with patch.object(cli, "_make_bus", return_value=EventBus()):
        with patch.object(cli, "_make_procedural_store", return_value=store):
            stdout, _ = run_cli(
                [
                    "cli",
                    "store-procedure",
                    "Deploy to Lambda",
                    "--steps",
                    "Package dependencies",
                    "Run sam deploy",
                    "--preconditions",
                    "AWS CLI configured",
                ]
            )

    assert "Stored procedure [" in stdout
    assert len(store.records) == 1
    record = store.records[0]
    assert isinstance(record, ProceduralMemory)
    assert record.content == "Deploy to Lambda"
    assert record.steps == ["Package dependencies", "Run sam deploy"]
    assert record.preconditions == ["AWS CLI configured"]
    assert record.modality == "text"
    print("  PASS  CLI stores text-backed procedural memories")


def test_store_media_backed_procedure_cli_smoke():
    store = RecordingProceduralStore()
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="cli_procedure_")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(b"%PDF-1.4\nrunbook")

    try:
        with tempfile.TemporaryDirectory(prefix="cli_media_root_") as media_root:
            with patch.object(cli, "_make_bus", return_value=EventBus()):
                    with patch.object(cli, "_make_procedural_store", return_value=store):
                        with patch.object(cli, "_make_media_store", return_value=MediaStore(media_root)):
                            stdout, _ = run_cli(
                                [
                                    "cli",
                                "store-procedure",
                                "Review migration checklist",
                                "--steps",
                                "Open the checklist",
                                "Run the migration",
                                "--file",
                                path,
                                "--modality",
                                "multimodal",
                                "--media-type",
                                "pdf",
                                "--text-description",
                                    "Reference PDF for the migration sequence",
                                ]
                            )

                        assert "Stored procedure [" in stdout
                        record = store.records[0]
                        assert record.modality == "multimodal"
                        assert record.media_type == "pdf"
                        assert record.media_ref == str(Path(media_root) / "documents" / f"{record.id}.pdf")
                        assert Path(record.media_ref).read_bytes() == b"%PDF-1.4\nrunbook"
                        assert record.text_description == "Reference PDF for the migration sequence"
                        print("  PASS  CLI stores media-backed procedures with owned supporting media")
    finally:
        os.remove(path)


def test_record_outcome_cli_updates_counts():
    record = ProceduralMemory(content="Deploy to Lambda", steps=["Package", "Deploy"])
    store = RecordingProceduralStore(record=record)

    with patch.object(cli, "_make_bus", return_value=EventBus()):
        with patch.object(cli, "_make_procedural_store", return_value=store):
            stdout, _ = run_cli(["cli", "record-outcome", record.id, "--failure"])

    assert store.outcomes == [(record.id, False)]
    assert "failure=1" in stdout
    print("  PASS  CLI record-outcome updates procedural counters")


def test_best_procedure_cli_prints_ranked_matches():
    record = ProceduralMemory(
        content="Deploy to Lambda via SAM",
        steps=["Package dependencies", "Run sam deploy"],
        success_count=3,
        failure_count=1,
    )
    store = RecordingProceduralStore(
        matches=[
            ProceduralMatch(
                record=record,
                similarity=0.9,
                wilson_score=record.wilson_score,
                combined_score=0.9 * 0.5 + record.wilson_score * 0.5,
            )
        ]
    )

    with patch.object(cli, "_make_bus", return_value=EventBus()):
        with patch.object(cli, "_make_procedural_store", return_value=store):
            stdout, _ = run_cli(["cli", "best-procedure", "deploy to Lambda", "-k", "1"])

    assert "Deploy to Lambda via SAM" in stdout
    assert "similarity=" in stdout
    assert "wilson=" in stdout
    print("  PASS  CLI best-procedure prints procedural ranking details")


if __name__ == "__main__":
    print("CLI tests:\n")
    test_store_episode_text_cli_smoke()
    test_store_episode_file_cli_smoke()
    test_store_semantic_image_cli_uses_required_content_and_owned_media()
    test_recent_cli_smoke()
    test_query_by_image_cli_embeds_file_and_prints_media_context()
    test_store_episode_multimodal_cli_infers_media_type_from_file()
    test_store_episode_file_cli_cleans_up_owned_media_on_failure()
    test_store_procedure_cli_smoke()
    test_store_media_backed_procedure_cli_smoke()
    test_record_outcome_cli_updates_counts()
    test_best_procedure_cli_prints_ranked_matches()
    print("\nAll tests passed.")
