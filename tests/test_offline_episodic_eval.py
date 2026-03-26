"""Deterministic offline evaluation harness for episodic-memory behavior."""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from retrieval.retriever import UnifiedRetriever
from stores.episodic_store import EpisodicStore
from stores.semantic_store import SemanticStore
from tests.helpers import (
    DeterministicMultimodalEmbedder,
    HashingEmbedder,
    cleanup_dir,
    make_temp_chroma_dir,
)


@dataclass(frozen=True)
class EvalResult:
    name: str
    expected: object
    actual: object

    @property
    def passed(self) -> bool:
        return self.actual == self.expected

    def failure_message(self) -> str:
        return f"{self.name}: expected {self.expected!r}, got {self.actual!r}"


@contextmanager
def evaluation_env(prefix: str, *, embedder=None):
    db_path = make_temp_chroma_dir(prefix)
    original_db_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = db_path
    active_embedder = embedder or HashingEmbedder()
    try:
        semantic_store = SemanticStore(embedder=active_embedder)
        episodic_store = EpisodicStore(embedder=active_embedder)
        retriever = UnifiedRetriever(
            stores={"semantic": semantic_store, "episodic": episodic_store}
        )
        yield Path(db_path), semantic_store, episodic_store, retriever
    finally:
        config.CHROMA_DB_PATH = original_db_path
        cleanup_dir(db_path)


def fixed_fixture_mixed_retrieval_scenario() -> EvalResult:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with evaluation_env("chroma_eval_mixed_") as (_, semantic_store, episodic_store, retriever):
        semantic_store.store(
            SemanticMemory(
                content="The rollback runbook covers migration recovery and worker restarts",
                created_at=now,
                importance=0.3,
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="We rolled back the billing migration after the deployment failed",
                session_id="session-mixed",
                created_at=now,
                importance=0.9,
            )
        )

        results = retriever.query("billing migration rollback", top_k=2)
        actual = [(result.record.memory_type, result.record.content) for result in results]

    return EvalResult(
        name="fixed-fixture mixed retrieval scenario",
        expected=[
            ("episodic", "We rolled back the billing migration after the deployment failed"),
            ("semantic", "The rollback runbook covers migration recovery and worker restarts"),
        ],
        actual=actual,
    )


def fixed_fixture_temporal_recall_scenario() -> EvalResult:
    base = datetime(2026, 1, 3, 9, 0, tzinfo=timezone.utc)
    with evaluation_env("chroma_eval_temporal_") as (_, _, episodic_store, retriever):
        episodic_store.store(
            EpisodicMemory(
                content="Opened the debugging session",
                session_id="session-temporal",
                created_at=base,
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Paired on retrieval weights",
                session_id="session-temporal",
                created_at=base + timedelta(minutes=10),
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Captured postmortem notes",
                session_id="session-temporal",
                created_at=base + timedelta(minutes=20),
            )
        )

        results = retriever.query_time_range(
            base + timedelta(minutes=5),
            base + timedelta(minutes=20),
        )
        actual = [record.content for record in results]

    return EvalResult(
        name="fixed-fixture temporal recall scenario",
        expected=[
            "Paired on retrieval weights",
            "Captured postmortem notes",
        ],
        actual=actual,
    )


def fixed_fixture_session_reconstruction_scenario() -> EvalResult:
    base = datetime(2026, 1, 4, 15, 0, tzinfo=timezone.utc)
    with evaluation_env("chroma_eval_session_") as (_, _, episodic_store, _):
        episodic_store.store(
            EpisodicMemory(
                content="User reported a failing retriever test",
                session_id="session-alpha",
                turn_number=1,
                created_at=base,
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Unrelated beta session note",
                session_id="session-beta",
                turn_number=1,
                created_at=base + timedelta(minutes=1),
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Agent isolated the Chroma persistence bug",
                session_id="session-alpha",
                turn_number=2,
                created_at=base + timedelta(minutes=2),
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="User confirmed the fix after rerunning the suite",
                session_id="session-alpha",
                turn_number=3,
                created_at=base + timedelta(minutes=3),
            )
        )

        results = episodic_store.get_by_session("session-alpha")
        actual = [record.content for record in results]

    return EvalResult(
        name="fixed-fixture session reconstruction scenario",
        expected=[
            "User reported a failing retriever test",
            "Agent isolated the Chroma persistence bug",
            "User confirmed the fix after rerunning the suite",
        ],
        actual=actual,
    )


def fixed_fixture_recent_event_scenario() -> EvalResult:
    base = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)
    with evaluation_env("chroma_eval_recent_") as (_, _, episodic_store, retriever):
        episodic_store.store(
            EpisodicMemory(
                content="Documented the regression harness",
                session_id="session-alpha",
                created_at=base,
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Captured screenshot of the failing job",
                session_id="session-beta",
                created_at=base + timedelta(minutes=10),
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Shipped the CLI patch",
                session_id="session-gamma",
                created_at=base + timedelta(minutes=20),
            )
        )

        results = retriever.query_recent(2)
        actual = [(record.session_id, record.content) for record in results]

    return EvalResult(
        name="fixed-fixture recent-event scenario",
        expected=[
            ("session-gamma", "Shipped the CLI patch"),
            ("session-beta", "Captured screenshot of the failing job"),
        ],
        actual=actual,
    )


def fixed_fixture_cross_modal_retrieval_scenario() -> EvalResult:
    embedder = DeterministicMultimodalEmbedder()
    with evaluation_env("chroma_eval_multimodal_", embedder=embedder) as (
        db_path,
        _,
        episodic_store,
        retriever,
    ):
        whiteboard_path = db_path / "whiteboard.png"
        whiteboard_path.write_bytes(b"whiteboard architecture retrieval timeline arrows")

        celebratory_path = db_path / "celebration.png"
        celebratory_path.write_bytes(b"confetti cake applause")

        episodic_store.store(
            EpisodicMemory(
                content="Whiteboard photo from the retrieval design session",
                session_id="session-multimodal",
                modality="image",
                media_ref=str(whiteboard_path),
                source_mime_type="image/png",
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Celebration photo from the launch party",
                session_id="session-multimodal",
                modality="image",
                media_ref=str(celebratory_path),
                source_mime_type="image/png",
            )
        )

        results = retriever.query("retrieval architecture whiteboard", top_k=1)
        actual = [
            (result.record.modality, result.record.content, Path(result.record.media_ref).name)
            for result in results
        ]

    return EvalResult(
        name="fixed-fixture cross-modal retrieval scenario",
        expected=[
            ("image", "Whiteboard photo from the retrieval design session", "whiteboard.png")
        ],
        actual=actual,
    )


def fixed_fixture_negative_no_episode_scenario() -> EvalResult:
    base = datetime(2026, 1, 6, 11, 0, tzinfo=timezone.utc)
    with evaluation_env("chroma_eval_negative_") as (_, _, episodic_store, retriever):
        episodic_store.store(
            EpisodicMemory(
                content="Finished the deploy checklist",
                session_id="session-negative",
                created_at=base,
            )
        )
        episodic_store.store(
            EpisodicMemory(
                content="Closed the incident channel",
                session_id="session-negative",
                created_at=base + timedelta(minutes=5),
            )
        )

        results = retriever.query_time_range(
            base + timedelta(hours=1),
            base + timedelta(hours=2),
        )
        actual = [record.content for record in results]

    return EvalResult(
        name="negative case where no episodic result should be returned",
        expected=[],
        actual=actual,
    )


def _assert_eval(result: EvalResult) -> None:
    assert result.passed, result.failure_message()
    print(f"  PASS  {result.name}")


def run_offline_episodic_eval() -> list[EvalResult]:
    return [
        fixed_fixture_mixed_retrieval_scenario(),
        fixed_fixture_temporal_recall_scenario(),
        fixed_fixture_session_reconstruction_scenario(),
        fixed_fixture_recent_event_scenario(),
        fixed_fixture_cross_modal_retrieval_scenario(),
        fixed_fixture_negative_no_episode_scenario(),
    ]


def test_fixed_fixture_mixed_retrieval_scenario():
    _assert_eval(fixed_fixture_mixed_retrieval_scenario())


def test_fixed_fixture_temporal_recall_scenario():
    _assert_eval(fixed_fixture_temporal_recall_scenario())


def test_fixed_fixture_session_reconstruction_scenario():
    _assert_eval(fixed_fixture_session_reconstruction_scenario())


def test_fixed_fixture_recent_event_scenario():
    _assert_eval(fixed_fixture_recent_event_scenario())


def test_fixed_fixture_cross_modal_retrieval_scenario():
    _assert_eval(fixed_fixture_cross_modal_retrieval_scenario())


def test_fixed_fixture_negative_no_episode_scenario():
    _assert_eval(fixed_fixture_negative_no_episode_scenario())


if __name__ == "__main__":
    print("Offline episodic-memory evaluation harness:\n")
    failures = []
    for result in run_offline_episodic_eval():
        if result.passed:
            print(f"  PASS  {result.name}")
        else:
            print(f"  FAIL  {result.failure_message()}")
            failures.append(result)

    if failures:
        print(f"\n{len(failures)} evaluation(s) failed.")
        raise SystemExit(1)

    print("\nAll evaluations passed.")
