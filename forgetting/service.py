from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from events.bus import EventBus
from forgetting.contradiction import ContradictionDetector
from forgetting.decay import compute_decay_score
from models.base import MemoryRecord
from models.procedural import ProceduralMemory
from stores.base import BaseStore
from stores.media_store import MediaStore

_ACTION_KEEP = "keep"
_ACTION_FADE = "fade"
_ACTION_PRUNE = "prune"

_REASON_TIME_DECAY = "time_decay"
_REASON_SUPERSEDED = "superseded"
_REASON_LIKELY_DUPLICATE = "likely_duplicate"
_REASON_LOW_PERFORMANCE = "low_performance"

_REASON_PRIORITY = {
    _REASON_SUPERSEDED: 0,
    _REASON_LIKELY_DUPLICATE: 1,
    _REASON_LOW_PERFORMANCE: 2,
    _REASON_TIME_DECAY: 3,
}

_STORE_TYPES = ("semantic", "episodic", "procedural")


@dataclass(slots=True)
class ForgettingDecision:
    record_id: str
    memory_type: str
    action: str
    reason: str | None
    score: float
    media_deleted: bool = False
    executed: bool = False
    record_skip_reason: str | None = None
    media_skip_reason: str | None = None
    old_importance: float | None = None
    new_importance: float | None = None


@dataclass(slots=True)
class ForgettingReport:
    decisions: list[ForgettingDecision]
    dry_run: bool
    duplicates_flagged: int = 0
    scanned: int = 0
    kept: int = 0
    faded: int = 0
    pruned: int = 0
    media_deleted: int = 0
    skipped_records: int = 0
    skipped_media: int = 0
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scanned = len(self.decisions)
        self.duplicates_flagged = sum(
            1
            for decision in self.decisions
            if decision.reason == _REASON_LIKELY_DUPLICATE and decision.action == _ACTION_PRUNE
        )
        self.media_deleted = sum(1 for decision in self.decisions if decision.media_deleted)
        self.skipped_records = sum(
            1 for decision in self.decisions if decision.record_skip_reason == "missing_record"
        )
        self.skipped_media = sum(
            1 for decision in self.decisions if decision.media_skip_reason == "missing_media"
        )

        counts = {
            memory_type: {_ACTION_KEEP: 0, _ACTION_FADE: 0, _ACTION_PRUNE: 0}
            for memory_type in _STORE_TYPES
        }

        for decision in self.decisions:
            if decision.memory_type not in counts:
                counts[decision.memory_type] = {
                    _ACTION_KEEP: 0,
                    _ACTION_FADE: 0,
                    _ACTION_PRUNE: 0,
                }

            if decision.action == _ACTION_KEEP:
                counts[decision.memory_type][_ACTION_KEEP] += 1
                continue

            if self.dry_run or decision.executed:
                counts[decision.memory_type][decision.action] += 1

        self.kept = sum(bucket[_ACTION_KEEP] for bucket in counts.values())
        self.faded = sum(bucket[_ACTION_FADE] for bucket in counts.values())
        self.pruned = sum(bucket[_ACTION_PRUNE] for bucket in counts.values())
        self.by_type = counts

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "scanned": self.scanned,
            "kept": self.kept,
            "faded": self.faded,
            "pruned": self.pruned,
            "media_deleted": self.media_deleted,
            "duplicates_flagged": self.duplicates_flagged,
            "skipped_records": self.skipped_records,
            "skipped_media": self.skipped_media,
            "by_type": self.by_type,
            "decisions": [asdict(decision) for decision in self.decisions],
        }


@dataclass(slots=True)
class _PlannedDecision:
    record: MemoryRecord
    action: str
    reason: str | None
    score: float


class ForgettingService:
    def __init__(
        self,
        semantic_store: BaseStore,
        episodic_store: BaseStore,
        procedural_store: BaseStore,
        media_store: MediaStore,
        event_bus: EventBus | None = None,
        contradiction_detector: ContradictionDetector | None = None,
    ):
        self._stores = {
            "semantic": semantic_store,
            "episodic": episodic_store,
            "procedural": procedural_store,
        }
        self._media_store = media_store
        self._event_bus = event_bus
        self._duplicate_detector = contradiction_detector or ContradictionDetector(semantic_store)

    def run_cycle(self, dry_run: bool = False) -> ForgettingReport:
        cycle_now = datetime.now(timezone.utc)
        planned_decisions = self._plan_cycle(cycle_now)
        decisions = [self._to_report_decision(plan) for plan in planned_decisions]

        if not dry_run:
            self._execute_fades(planned_decisions, decisions)
            self._execute_prunes(planned_decisions, decisions)

        report = ForgettingReport(decisions=decisions, dry_run=dry_run)
        self._emit_event(
            "forgetting.cycle_dry_run" if dry_run else "forgetting.cycle_completed",
            report.to_event_payload(),
        )
        return report

    def _plan_cycle(self, cycle_now: datetime) -> list[_PlannedDecision]:
        all_records = self._scan_records()
        scores = {
            record.id: compute_decay_score(record, now=cycle_now)
            for record in all_records.values()
        }
        active_superseded_ids = self._active_superseded_ids(all_records)
        duplicate_prune_ids = self._duplicate_prune_ids(all_records, active_superseded_ids)

        planned: list[_PlannedDecision] = []
        for record in all_records.values():
            reason = self._select_reason(
                record,
                score=scores[record.id],
                active_superseded_ids=active_superseded_ids,
                duplicate_prune_ids=duplicate_prune_ids,
            )
            planned.append(
                _PlannedDecision(
                    record=record,
                    action=self._action_for_reason(
                        reason,
                        record,
                        score=scores[record.id],
                    ),
                    reason=reason,
                    score=scores[record.id],
                )
            )

        return sorted(planned, key=lambda item: (item.record.memory_type, item.record.id))

    def _scan_records(self) -> dict[str, MemoryRecord]:
        records: dict[str, MemoryRecord] = {}
        for store in self._stores.values():
            for record in store.get_all_records(include_embeddings=True):
                records[record.id] = record
        return records

    def _active_superseded_ids(self, all_records: dict[str, MemoryRecord]) -> set[str]:
        active: set[str] = set()
        for record in all_records.values():
            successor_id = getattr(record, "superseded_by", None)
            if not successor_id:
                continue
            successor = all_records.get(successor_id)
            if successor is None:
                try:
                    successor = self._stores[record.memory_type].get_by_id(successor_id)
                except KeyError:
                    successor = None
            if successor is not None:
                active.add(record.id)
        return active

    def _duplicate_prune_ids(
        self,
        all_records: dict[str, MemoryRecord],
        active_superseded_ids: set[str],
    ) -> set[str]:
        pairs = self._duplicate_detector.find_likely_duplicates_batch(
            threshold=config.SEMANTIC_DUPLICATE_THRESHOLD
        )
        if not pairs:
            return set()

        parents: dict[str, str] = {}

        def find(record_id: str) -> str:
            parents.setdefault(record_id, record_id)
            if parents[record_id] != record_id:
                parents[record_id] = find(parents[record_id])
            return parents[record_id]

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left_id, right_id, _similarity in pairs:
            if left_id not in all_records or right_id not in all_records:
                continue
            union(left_id, right_id)

        components: dict[str, list[MemoryRecord]] = {}
        for record_id in list(parents):
            root = find(record_id)
            components.setdefault(root, []).append(all_records[record_id])

        prune_ids: set[str] = set()
        for records in components.values():
            if len(records) < 2:
                continue
            winner = max(
                records,
                key=lambda record: self._duplicate_winner_key(record, active_superseded_ids),
            )
            for record in records:
                if record.id != winner.id and record.id not in active_superseded_ids:
                    prune_ids.add(record.id)
        return prune_ids

    def _duplicate_winner_key(
        self,
        record: MemoryRecord,
        active_superseded_ids: set[str],
    ) -> tuple[int, float, int, float, str]:
        created_at = record.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)

        return (
            0 if record.id not in active_superseded_ids else -1,
            float(getattr(record, "importance", 0.0)),
            int(getattr(record, "access_count", 0)),
            created_at.timestamp(),
            # IDs are UUID-like ASCII strings in this codebase, so inverting each
            # byte gives us a deterministic descending fallback while still using
            # Python's normal tuple comparison.
            self._descending_string_key(record.id),
        )

    def _descending_string_key(self, value: str) -> str:
        return "".join(chr(255 - ord(char)) for char in value)

    def _action_for_reason(
        self,
        reason: str | None,
        record: MemoryRecord,
        *,
        score: float,
    ) -> str:
        if reason in {_REASON_SUPERSEDED, _REASON_LIKELY_DUPLICATE}:
            return _ACTION_PRUNE

        if reason == _REASON_LOW_PERFORMANCE:
            return (
                _ACTION_PRUNE
                if self._is_prune_level_low_performance(record)
                else _ACTION_FADE
            )

        prune_threshold, fade_threshold = self._thresholds_for(record.memory_type)
        if score < prune_threshold:
            return _ACTION_PRUNE
        if score < fade_threshold:
            return _ACTION_FADE
        return _ACTION_KEEP

    def _select_reason(
        self,
        record: MemoryRecord,
        *,
        score: float,
        active_superseded_ids: set[str],
        duplicate_prune_ids: set[str],
    ) -> str | None:
        candidates: list[str] = []
        if record.id in active_superseded_ids:
            candidates.append(_REASON_SUPERSEDED)
        if record.id in duplicate_prune_ids:
            candidates.append(_REASON_LIKELY_DUPLICATE)
        if self._is_low_performance(record):
            candidates.append(_REASON_LOW_PERFORMANCE)

        prune_threshold, fade_threshold = self._thresholds_for(record.memory_type)
        if score < fade_threshold:
            candidates.append(_REASON_TIME_DECAY)
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: _REASON_PRIORITY[candidate])

    def _thresholds_for(self, memory_type: str) -> tuple[float, float]:
        if memory_type == "semantic":
            return config.SEMANTIC_PRUNE_THRESHOLD, config.SEMANTIC_FADE_THRESHOLD
        if memory_type == "episodic":
            return config.EPISODIC_PRUNE_THRESHOLD, config.EPISODIC_FADE_THRESHOLD
        if memory_type == "procedural":
            return config.PROCEDURAL_PRUNE_THRESHOLD, config.PROCEDURAL_FADE_THRESHOLD
        raise ValueError(f"Unsupported memory_type '{memory_type}'")

    def _is_low_performance(self, record: MemoryRecord) -> bool:
        if not isinstance(record, ProceduralMemory):
            return False
        if record.total_outcomes == 0:
            return False
        return record.wilson_score < config.PROCEDURAL_LOW_PERF_WILSON_THRESHOLD

    def _is_prune_level_low_performance(self, record: MemoryRecord) -> bool:
        if not isinstance(record, ProceduralMemory):
            return False
        return (
            self._is_low_performance(record)
            and record.total_outcomes >= config.PROCEDURAL_LOW_PERF_MIN_OUTCOMES
        )

    def _to_report_decision(self, planned: _PlannedDecision) -> ForgettingDecision:
        old_importance = float(getattr(planned.record, "importance", 0.0))
        new_importance = None
        if planned.action == _ACTION_FADE:
            new_importance = max(config.FADE_FLOOR, old_importance * config.FADE_FACTOR)
        return ForgettingDecision(
            record_id=planned.record.id,
            memory_type=planned.record.memory_type,
            action=planned.action,
            reason=planned.reason,
            score=planned.score,
            old_importance=old_importance,
            new_importance=new_importance,
        )

    def _execute_fades(
        self,
        planned: list[_PlannedDecision],
        decisions: list[ForgettingDecision],
    ) -> None:
        by_id = {decision.record_id: decision for decision in decisions}
        for item in planned:
            if item.action != _ACTION_FADE:
                continue
            decision = by_id[item.record.id]
            store = self._stores[item.record.memory_type]
            current = store.get_by_id(item.record.id)
            if current is None:
                decision.record_skip_reason = "missing_record"
                continue

            current.importance = decision.new_importance if decision.new_importance is not None else 0.0
            store.replace(current)
            decision.executed = True
            self._emit_event(
                "memory.faded",
                {
                    "record_id": decision.record_id,
                    "memory_type": decision.memory_type,
                    "reason": decision.reason,
                    "score": decision.score,
                    "old_importance": decision.old_importance,
                    "new_importance": decision.new_importance,
                },
            )

    def _execute_prunes(
        self,
        planned: list[_PlannedDecision],
        decisions: list[ForgettingDecision],
    ) -> None:
        prune_items = [item for item in planned if item.action == _ACTION_PRUNE]
        by_id = {decision.record_id: decision for decision in decisions}

        for chunk in self._iter_chunks(prune_items, 10):
            for item in chunk:
                store = self._stores[item.record.memory_type]
                decision = by_id[item.record.id]
                current = store.get_by_id(item.record.id)
                if current is None:
                    decision.record_skip_reason = "missing_record"
                    continue

                had_media = bool(current.media_ref)
                media_ref = current.media_ref
                delete_media = bool(
                    media_ref and self._media_store.owns(media_ref) and Path(media_ref).exists()
                )

                store.delete(item.record.id)
                decision.executed = True

                if media_ref and self._media_store.owns(media_ref):
                    if delete_media:
                        self._media_store.delete(media_ref)
                        decision.media_deleted = True
                    else:
                        decision.media_skip_reason = "missing_media"

                self._emit_event(
                    "memory.forgotten",
                    {
                        "record_id": decision.record_id,
                        "memory_type": decision.memory_type,
                        "reason": decision.reason,
                        "score": decision.score,
                        "had_media": had_media,
                        **({"media_ref": media_ref} if media_ref else {}),
                    },
                )

    def _iter_chunks(self, items: list[_PlannedDecision], chunk_size: int) -> list[list[_PlannedDecision]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(event_type, data)
