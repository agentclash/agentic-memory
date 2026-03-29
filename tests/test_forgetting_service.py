"""Verify forgetting-cycle planning, execution, and event contracts."""

import os
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.bus import EventBus
from forgetting.service import ForgettingService
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from stores.base import BaseStore
from stores.media_store import MediaStore


class FakeDuplicateDetector:
    def __init__(self, pairs=None):
        self._pairs = pairs or []

    def find_likely_duplicates_batch(self, threshold: float = 0.95):
        return list(self._pairs)


class FakeStore(BaseStore):
    def __init__(self, records=None):
        super().__init__()
        self._records = {record.id: record for record in (records or [])}
        self.replace_calls = []
        self.delete_calls = []

    def store(self, record):
        self._records[record.id] = record
        return record.id

    def get_by_id(self, record_id):
        return self._records.get(record_id)

    def get_all_records(self, include_embeddings: bool = False):
        records = list(self._records.values())
        if include_embeddings:
            return records
        return [replace(record, embedding=None) for record in records]

    def retrieve(self, query: str, top_k: int = 5):
        return []

    def retrieve_by_vector(self, vector: list[float], top_k: int = 5):
        return []

    def update_access(self, record_id: str) -> None:
        return None

    def delete(self, record_id: str) -> None:
        self.delete_calls.append(record_id)
        self._records.pop(record_id, None)

    def replace(self, record) -> None:
        self.replace_calls.append(record.id)
        self._records[record.id] = record


class EventRecorder:
    def __init__(self, bus: EventBus, *event_types: str):
        self.events = []
        for event_type in event_types:
            bus.subscribe(event_type, self.events.append)


def _media_store():
    root = tempfile.mkdtemp(prefix="forgetting_media_")
    return MediaStore(root), root


def _make_service(
    semantic_records=None,
    episodic_records=None,
    procedural_records=None,
    *,
    duplicate_pairs=None,
    event_bus=None,
):
    media_store, media_root = _media_store()
    service = ForgettingService(
        semantic_store=FakeStore(semantic_records),
        episodic_store=FakeStore(episodic_records),
        procedural_store=FakeStore(procedural_records),
        media_store=media_store,
        event_bus=event_bus,
        contradiction_detector=FakeDuplicateDetector(duplicate_pairs),
    )
    return service, media_root


def _ids_for(report):
    return {decision.record_id: decision for decision in report.decisions}


def test_dry_run_resolves_duplicate_clusters_component_wise(monkeypatch):
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    first = SemanticMemory(
        id="semantic-a",
        content="A",
        importance=0.7,
        access_count=2,
        created_at=now - timedelta(days=30),
    )
    second = SemanticMemory(
        id="semantic-b",
        content="B",
        importance=0.9,
        access_count=1,
        created_at=now - timedelta(days=10),
    )
    third = SemanticMemory(
        id="semantic-c",
        content="C",
        importance=0.8,
        access_count=5,
        created_at=now - timedelta(days=1),
    )
    unrelated = SemanticMemory(id="semantic-d", content="D", importance=0.6, created_at=now)

    monkeypatch.setattr(
        "forgetting.service.compute_decay_score",
        lambda record, now=None: {
            "semantic-a": 0.9,
            "semantic-b": 0.9,
            "semantic-c": 0.9,
            "semantic-d": 0.9,
        }[record.id],
    )

    bus = EventBus()
    recorder = EventRecorder(bus, "forgetting.cycle_dry_run", "memory.forgotten", "memory.faded")
    service, media_root = _make_service(
        semantic_records=[first, second, third, unrelated],
        duplicate_pairs=[
            ("semantic-a", "semantic-b", 0.99),
            ("semantic-b", "semantic-c", 0.98),
        ],
        event_bus=bus,
    )

    try:
        report = service.run_cycle(dry_run=True)
        decisions = _ids_for(report)

        assert decisions["semantic-b"].action == "keep"
        assert decisions["semantic-a"].action == "prune"
        assert decisions["semantic-c"].action == "prune"
        assert decisions["semantic-a"].reason == "likely_duplicate"
        assert decisions["semantic-c"].reason == "likely_duplicate"
        assert report.duplicates_flagged == 2
        assert report.pruned == 2
        assert report.kept == 2
        assert [event.event_type for event in recorder.events] == ["forgetting.cycle_dry_run"]
        assert service._stores["semantic"].get_by_id("semantic-a") is not None
        print("  PASS  dry run resolves transitive duplicate clusters to one survivor without mutation")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_duplicate_tiebreaker_falls_through_access_recency_then_id(monkeypatch):
    first = SemanticMemory(
        id="semantic-a",
        content="A",
        importance=0.8,
        access_count=3,
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    second = SemanticMemory(
        id="semantic-b",
        content="B",
        importance=0.8,
        access_count=5,
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    third = SemanticMemory(
        id="semantic-c",
        content="C",
        importance=0.8,
        access_count=5,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    fourth = SemanticMemory(
        id="semantic-d",
        content="D",
        importance=0.8,
        access_count=5,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.9)
    service, media_root = _make_service(
        semantic_records=[first, second, third, fourth],
        duplicate_pairs=[
            ("semantic-a", "semantic-b", 0.99),
            ("semantic-b", "semantic-c", 0.99),
            ("semantic-c", "semantic-d", 0.99),
        ],
    )

    try:
        report = service.run_cycle(dry_run=True)
        decisions = _ids_for(report)

        assert decisions["semantic-c"].action == "keep"
        assert decisions["semantic-a"].action == "prune"
        assert decisions["semantic-b"].action == "prune"
        assert decisions["semantic-d"].action == "prune"
        print("  PASS  duplicate winner selection falls through access, recency, then id deterministically")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_supersession_forces_prune_only_when_successor_exists(monkeypatch):
    old = SemanticMemory(
        id="old",
        content="Old fact",
        importance=0.9,
        superseded_by="kept",
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    kept = SemanticMemory(
        id="kept",
        content="New fact",
        importance=0.9,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    missing_successor = SemanticMemory(
        id="maybe-old",
        content="Maybe old",
        importance=0.9,
        superseded_by="gone",
        created_at=datetime(2026, 3, 3, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        "forgetting.service.compute_decay_score",
        lambda record, now=None: 0.95,
    )

    service, media_root = _make_service(semantic_records=[old, kept, missing_successor])

    try:
        report = service.run_cycle(dry_run=True)
        decisions = _ids_for(report)

        assert decisions["old"].action == "prune"
        assert decisions["old"].reason == "superseded"
        assert decisions["maybe-old"].action == "keep"
        assert decisions["maybe-old"].reason is None
        print("  PASS  supersession forced-prunes only when the replacement record still exists")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_real_run_stages_fade_and_prune_then_deletes_owned_media(monkeypatch):
    fade_record = SemanticMemory(
        id="fade-me",
        content="Fade me",
        importance=0.8,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    prune_record = EpisodicMemory(
        id="prune-me",
        content="Prune me",
        session_id="session-1",
        importance=0.4,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    bus = EventBus()
    recorder = EventRecorder(bus, "memory.faded", "memory.forgotten", "forgetting.cycle_completed")
    service, media_root = _make_service(
        semantic_records=[fade_record],
        episodic_records=[prune_record],
        event_bus=bus,
    )
    owned_media = service._media_store.store_bytes(b"audio", "clip.mp3", prune_record.id)
    prune_record.media_ref = owned_media

    monkeypatch.setattr(
        "forgetting.service.compute_decay_score",
        lambda record, now=None: {
            "fade-me": 0.15,
            "prune-me": 0.05,
        }[record.id],
    )

    try:
        report = service.run_cycle(dry_run=False)
        decisions = _ids_for(report)

        assert service._stores["semantic"].replace_calls == ["fade-me"]
        assert service._stores["episodic"].delete_calls == ["prune-me"]
        assert service._stores["semantic"].get_by_id("fade-me").importance == pytest.approx(0.4)
        assert service._stores["episodic"].get_by_id("prune-me") is None
        assert not os.path.exists(owned_media)
        assert decisions["fade-me"].executed is True
        assert decisions["prune-me"].executed is True
        assert decisions["prune-me"].media_deleted is True
        assert report.faded == 1
        assert report.pruned == 1
        assert report.media_deleted == 1
        assert [event.event_type for event in recorder.events] == [
            "memory.faded",
            "memory.forgotten",
            "forgetting.cycle_completed",
        ]
        assert recorder.events[0].data["record_id"] == "fade-me"
        assert recorder.events[0].data["reason"] == "time_decay"
        assert recorder.events[0].data["old_importance"] == 0.8
        assert recorder.events[0].data["new_importance"] == pytest.approx(0.4)
        assert recorder.events[1].data["record_id"] == "prune-me"
        assert recorder.events[1].data["reason"] == "time_decay"
        assert recorder.events[1].data["had_media"] is True
        assert recorder.events[2].data["pruned"] == 1
        assert recorder.events[2].data["media_deleted"] == 1
        print("  PASS  real run applies staged fades and prunes before deleting owned media")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_procedural_low_performance_forces_fade_or_prune(monkeypatch):
    forced_fade = ProceduralMemory(
        id="proc-fade",
        content="Weak but early",
        steps=["One"],
        success_count=0,
        failure_count=5,
        importance=0.8,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    forced_prune = ProceduralMemory(
        id="proc-prune",
        content="Statistically dead",
        steps=["One"],
        success_count=0,
        failure_count=12,
        importance=0.8,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.95)
    service, media_root = _make_service(procedural_records=[forced_fade, forced_prune])

    try:
        report = service.run_cycle(dry_run=True)
        decisions = _ids_for(report)

        assert decisions["proc-fade"].action == "fade"
        assert decisions["proc-fade"].reason == "low_performance"
        assert decisions["proc-prune"].action == "prune"
        assert decisions["proc-prune"].reason == "low_performance"
        print("  PASS  procedural low performance fades small samples and prunes statistically dead ones")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_zero_outcome_procedure_is_not_marked_low_performance(monkeypatch):
    untouched = ProceduralMemory(
        id="proc-untouched",
        content="Never tried yet",
        steps=["One"],
        importance=0.8,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.95)
    service, media_root = _make_service(procedural_records=[untouched])

    try:
        report = service.run_cycle(dry_run=True)
        decision = _ids_for(report)["proc-untouched"]

        assert decision.action == "keep"
        assert decision.reason is None
        print("  PASS  zero-outcome procedures are not treated as low-performance before they are ever tried")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_missing_record_is_tracked_as_skipped_not_pruned(monkeypatch):
    stale = SemanticMemory(
        id="missing-prune",
        content="Will vanish",
        importance=0.2,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.01)
    service, media_root = _make_service(semantic_records=[stale])
    service._stores["semantic"].get_by_id = lambda record_id: None  # type: ignore[method-assign]

    try:
        report = service.run_cycle(dry_run=False)
        decision = _ids_for(report)["missing-prune"]

        assert decision.executed is False
        assert decision.record_skip_reason == "missing_record"
        assert report.pruned == 0
        assert report.skipped_records == 1
        print("  PASS  records missing at delete time are reported as skipped instead of successful prunes")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_missing_record_is_tracked_as_skipped_for_fades_too(monkeypatch):
    record = SemanticMemory(
        id="missing-fade",
        content="Will vanish before fade",
        importance=0.8,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.15)
    service, media_root = _make_service(semantic_records=[record])
    service._stores["semantic"].get_by_id = lambda record_id: None  # type: ignore[method-assign]

    try:
        report = service.run_cycle(dry_run=False)
        decision = _ids_for(report)["missing-fade"]

        assert decision.executed is False
        assert decision.record_skip_reason == "missing_record"
        assert report.faded == 0
        assert report.skipped_records == 1
        print("  PASS  records missing at fade time are reported as skipped instead of being reinserted")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_iter_chunks_batches_prunes_in_groups_of_ten():
    service, media_root = _make_service()

    try:
        chunks = service._iter_chunks(list(range(21)), 10)
        assert [len(chunk) for chunk in chunks] == [10, 10, 1]
        print("  PASS  prune execution batching splits work into groups of ten")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_fade_floor_clamps_tiny_importance_values(monkeypatch):
    record = SemanticMemory(
        id="tiny-fade",
        content="Tiny fade",
        importance=0.001,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.15)
    service, media_root = _make_service(semantic_records=[record])

    try:
        report = service.run_cycle(dry_run=False)
        decision = _ids_for(report)["tiny-fade"]

        assert decision.new_importance == pytest.approx(0.01)
        assert service._stores["semantic"].get_by_id("tiny-fade").importance == pytest.approx(0.01)
        print("  PASS  fade floor prevents tiny importance values from collapsing below the configured minimum")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_one_cycle_applies_type_specific_thresholds_across_all_stores(monkeypatch):
    semantic = SemanticMemory(id="sem", content="semantic", importance=0.6)
    episodic = EpisodicMemory(id="epi", content="episodic", session_id="s", importance=0.6)
    procedural = ProceduralMemory(
        id="proc",
        content="procedural",
        steps=["One"],
        importance=0.6,
        success_count=10,
        failure_count=0,
    )

    monkeypatch.setattr(
        "forgetting.service.compute_decay_score",
        lambda record, now=None: {
            "sem": 0.15,
            "epi": 0.25,
            "proc": 0.35,
        }[record.id],
    )
    service, media_root = _make_service(
        semantic_records=[semantic],
        episodic_records=[episodic],
        procedural_records=[procedural],
    )

    try:
        report = service.run_cycle(dry_run=True)
        decisions = _ids_for(report)

        assert decisions["sem"].action == "fade"
        assert decisions["epi"].action == "fade"
        assert decisions["proc"].action == "keep"
        assert report.by_type["semantic"]["fade"] == 1
        assert report.by_type["episodic"]["fade"] == 1
        assert report.by_type["procedural"]["keep"] == 1
        print("  PASS  one cycle respects type-specific thresholds across semantic, episodic, and procedural stores")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


def test_supersession_reason_beats_duplicate_reason_when_both_apply(monkeypatch):
    superseded = SemanticMemory(
        id="old",
        content="Old fact",
        importance=0.9,
        superseded_by="new",
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    current = SemanticMemory(
        id="new",
        content="New fact",
        importance=0.8,
        created_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )

    monkeypatch.setattr("forgetting.service.compute_decay_score", lambda record, now=None: 0.95)
    service, media_root = _make_service(
        semantic_records=[superseded, current],
        duplicate_pairs=[("old", "new", 0.99)],
    )

    try:
        report = service.run_cycle(dry_run=True)
        decision = _ids_for(report)["old"]

        assert decision.action == "prune"
        assert decision.reason == "superseded"
        print("  PASS  supersession outranks duplicate handling when both apply to the same record")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
