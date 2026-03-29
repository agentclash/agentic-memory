from __future__ import annotations

from datetime import datetime, timezone

import config
from models.base import MemoryRecord


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _effective_half_life(record: MemoryRecord) -> float:
    memory_type = record.memory_type.lower()

    if memory_type == "semantic":
        half_life = config.SEMANTIC_HALF_LIFE_DAYS
    elif memory_type == "episodic":
        half_life = config.EPISODIC_HALF_LIFE_DAYS
    elif memory_type == "procedural":
        half_life = config.PROCEDURAL_HALF_LIFE_DAYS
    else:
        raise ValueError(f"Unsupported memory_type '{record.memory_type}'")

    importance = _clamp(float(getattr(record, "importance", 0.0)))
    # TODO(#49): Evaluate replacing this hard threshold with a smooth
    # importance-weighted half-life curve once issue #42 is shipped.
    # https://github.com/agentclash/agentic-memory/issues/49
    if importance >= config.IMPORTANCE_FLOOR_THRESHOLD:
        half_life *= config.IMPORTANCE_FLOOR_MULTIPLIER
    return half_life


def _days_since_last_access(record: MemoryRecord, now: datetime) -> float:
    last_accessed = record.last_accessed_at or record.created_at
    delta = _as_utc(now) - _as_utc(last_accessed)
    return max(0.0, delta.total_seconds() / 86400)


def compute_decay_score(record: MemoryRecord, now: datetime | None = None) -> float:
    current_time = _as_utc(now or datetime.now(timezone.utc))
    days_since_last_access = _days_since_last_access(record, current_time)
    half_life = _effective_half_life(record)
    time_factor = 0.5 ** (days_since_last_access / half_life)
    access_boost = min(1.0, max(0.0, float(record.access_count)) / config.ACCESS_NORMALIZATION_CONSTANT)
    importance = _clamp(float(getattr(record, "importance", 0.0)))

    memory_type = record.memory_type.lower()
    if memory_type in {"semantic", "episodic"}:
        score = time_factor * 0.5 + access_boost * 0.25 + importance * 0.25
    elif memory_type == "procedural":
        wilson_score = _clamp(float(getattr(record, "wilson_score", 0.0)))
        score = time_factor * 0.3 + wilson_score * 0.5 + access_boost * 0.2
    else:
        raise ValueError(f"Unsupported memory_type '{record.memory_type}'")

    return _clamp(score)
