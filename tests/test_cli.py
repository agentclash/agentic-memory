"""Verify CLI episodic commands without provider-backed dependencies."""

import io
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import cli
from events.bus import EventBus
from models.episodic import EpisodicMemory


class RecordingStore:
    def __init__(self):
        self.records = []

    def store(self, record):
        self.records.append(record)
        return record.id


class FakeRetriever:
    def __init__(self, recent_records=None):
        self.recent_records = recent_records or []
        self.recent_calls = []

    def query_recent(self, n: int):
        self.recent_calls.append(n)
        return self.recent_records[:n]


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
        with patch.object(cli, "_make_bus", return_value=EventBus()):
            with patch.object(cli, "_make_episodic_store", return_value=store):
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
        assert record.media_ref == os.path.abspath(path)
        assert record.source_mime_type == "image/png"
        print("  PASS  CLI stores file-backed episodic memories with modality metadata")
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


if __name__ == "__main__":
    print("CLI tests:\n")
    test_store_episode_text_cli_smoke()
    test_store_episode_file_cli_smoke()
    test_recent_cli_smoke()
    print("\nAll tests passed.")
