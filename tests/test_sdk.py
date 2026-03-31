"""Tests for the agentic_memory SDK (Memory class)."""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_memory import (
    Memory,
    RankedResult,
    ContradictionCandidate,
    ProceduralMatch,
    ForgettingReport,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
)
from tests.helpers import HashingEmbedder, make_temp_chroma_dir, cleanup_dir


@pytest.fixture()
def mem(tmp_path):
    chroma_dir = make_temp_chroma_dir("sdk_test_")
    m = Memory(
        chroma_path=chroma_dir,
        media_root=str(tmp_path / "media"),
        embedder=HashingEmbedder(dimensions=64),
        embedding_dimensions=64,
    )
    yield m
    cleanup_dir(chroma_dir)


class TestRemember:
    def test_remember_returns_id(self, mem: Memory):
        rid = mem.remember("the sky is blue")
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_remember_with_options(self, mem: Memory):
        rid = mem.remember(
            "deploy key rotates every 90 days",
            importance=0.9,
            category="infra",
            domain="security",
            confidence=0.95,
        )
        record = mem.get(rid, type="semantic")
        assert record is not None
        assert record.importance == 0.9
        assert record.category == "infra"
        assert record.domain == "security"

    def test_remember_episode_returns_id(self, mem: Memory):
        rid = mem.remember_episode("user debugged auth flow", session="s1")
        assert isinstance(rid, str)

    def test_remember_episode_with_session(self, mem: Memory):
        rid = mem.remember_episode("built the feature", session="sess-42", turn=3)
        record = mem.get(rid, type="episodic")
        assert record is not None
        assert record.session_id == "sess-42"
        assert record.turn_number == 3

    def test_remember_procedure_returns_id(self, mem: Memory):
        rid = mem.remember_procedure(
            "deploy to production",
            steps=["run tests", "build image", "push"],
        )
        assert isinstance(rid, str)

    def test_remember_procedure_with_preconditions(self, mem: Memory):
        rid = mem.remember_procedure(
            "rollback deployment",
            steps=["identify version", "kubectl rollout undo"],
            preconditions=["kubectl access"],
            importance=0.8,
        )
        record = mem.get(rid, type="procedural")
        assert record is not None
        assert record.preconditions == ["kubectl access"]
        assert record.importance == 0.8


class TestRecall:
    def test_recall_finds_stored_memory(self, mem: Memory):
        mem.remember("python is a programming language")
        results = mem.recall("programming language")
        assert len(results) > 0
        assert isinstance(results[0], RankedResult)
        assert results[0].record.content == "python is a programming language"

    def test_recall_with_type_filter(self, mem: Memory):
        mem.remember("fact about dogs")
        mem.remember_episode("walked the dog", session="s1")
        results = mem.recall("dog", types=["semantic"])
        assert all(r.record.memory_type == "semantic" for r in results)

    def test_recall_empty_returns_empty(self, mem: Memory):
        results = mem.recall("nonexistent topic")
        assert results == []

    def test_recall_respects_top_k(self, mem: Memory):
        for i in range(5):
            mem.remember(f"fact number {i} about testing")
        results = mem.recall("testing", top_k=2)
        assert len(results) <= 2


class TestRecallEpisodes:
    def test_recall_recent(self, mem: Memory):
        mem.remember_episode("first event", session="s1")
        mem.remember_episode("second event", session="s1")
        records = mem.recall_episodes(mode="recent", limit=1)
        assert len(records) == 1

    def test_recall_by_session(self, mem: Memory):
        mem.remember_episode("alpha event", session="alpha")
        mem.remember_episode("beta event", session="beta")
        records = mem.recall_episodes(mode="session", session="alpha")
        assert len(records) == 1
        assert records[0].content == "alpha event"

    def test_recall_session_requires_session_param(self, mem: Memory):
        with pytest.raises(ValueError, match="session is required"):
            mem.recall_episodes(mode="session")

    def test_recall_time_range(self, mem: Memory):
        now = datetime.now(timezone.utc)
        mem.remember_episode("timed event", session="s1")
        records = mem.recall_episodes(
            mode="time_range",
            start=now - timedelta(minutes=1),
            end=now + timedelta(minutes=1),
        )
        assert len(records) >= 1

    def test_recall_time_range_requires_start_end(self, mem: Memory):
        with pytest.raises(ValueError, match="start and end"):
            mem.recall_episodes(mode="time_range", start=datetime.now(timezone.utc))

    def test_unknown_mode_raises(self, mem: Memory):
        with pytest.raises(ValueError, match="Unknown mode"):
            mem.recall_episodes(mode="invalid")


class TestRecallProcedures:
    def test_recall_procedures(self, mem: Memory):
        mem.remember_procedure("deploy app", steps=["build", "push", "apply"])
        matches = mem.recall_procedures("deploy application")
        assert len(matches) > 0
        assert isinstance(matches[0], ProceduralMatch)


class TestContradictions:
    def test_find_contradictions(self, mem: Memory):
        id1 = mem.remember("the capital of France is Paris")
        id2 = mem.remember("the capital of France is Paris and it is lovely")
        candidates = mem.find_contradictions(id1)
        # With the hashing embedder, similar text produces similar vectors
        assert isinstance(candidates, list)

    def test_find_contradictions_missing_id_raises(self, mem: Memory):
        with pytest.raises(ValueError, match="not found"):
            mem.find_contradictions("nonexistent-id")

    def test_resolve_contradiction(self, mem: Memory):
        id1 = mem.remember("old fact")
        id2 = mem.remember("new fact replaces old")
        mem.resolve_contradiction(keep_id=id2, supersede_id=id1)
        old = mem.get(id1, type="semantic")
        assert old is not None
        assert old.superseded_by == id2
        assert old.importance == 0.0


class TestOutcomes:
    def test_record_outcome(self, mem: Memory):
        rid = mem.remember_procedure("test proc", steps=["step1"])
        mem.record_outcome(rid, success=True)
        mem.record_outcome(rid, success=True)
        mem.record_outcome(rid, success=False)
        record = mem.get(rid, type="procedural")
        assert record.success_count == 2
        assert record.failure_count == 1


class TestForgetting:
    def test_dry_run(self, mem: Memory):
        mem.remember("something to maybe forget")
        report = mem.forget(dry_run=True)
        assert isinstance(report, ForgettingReport)
        assert report.dry_run is True

    def test_run(self, mem: Memory):
        mem.remember("something to maybe forget")
        report = mem.forget()
        assert isinstance(report, ForgettingReport)
        assert report.dry_run is False


class TestDirectAccess:
    def test_get_semantic(self, mem: Memory):
        rid = mem.remember("fact")
        record = mem.get(rid, type="semantic")
        assert record is not None
        assert record.content == "fact"

    def test_get_unknown_type_raises(self, mem: Memory):
        with pytest.raises(ValueError, match="Unknown type"):
            mem.get("some-id", type="invalid")

    def test_get_missing_returns_none(self, mem: Memory):
        result = mem.get("nonexistent", type="semantic")
        assert result is None

    def test_overview(self, mem: Memory):
        mem.remember("a fact")
        mem.remember_episode("an event", session="s1")
        overview = mem.overview()
        assert overview["semantic"] == 1
        assert overview["episodic"] == 1
        assert overview["procedural"] == 0
        assert overview["total"] == 2

    def test_events(self, mem: Memory):
        mem.remember("trigger events")
        evts = mem.events()
        assert isinstance(evts, list)
        assert len(evts) > 0
        assert "event_type" in evts[0]

    def test_runtime_escape_hatch(self, mem: Memory):
        assert mem.runtime is not None
        assert hasattr(mem.runtime, "semantic_store")


class TestExports:
    """Verify that key types are importable from the SDK."""

    def test_all_exports(self):
        from agentic_memory import (
            Memory,
            MemoryRuntime,
            RankedResult,
            ContradictionCandidate,
            ProceduralMatch,
            ForgettingReport,
            SemanticMemory,
            EpisodicMemory,
            ProceduralMemory,
            MemoryRecord,
        )
        assert Memory is not None
        assert RankedResult is not None
