"""Verify forgetting decay scoring behavior for all memory types."""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgetting.decay import compute_decay_score
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory


def test_newer_episodic_memory_scores_higher_than_older_one():
    now = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
    recent = EpisodicMemory(
        content="Recent event",
        created_at=now - timedelta(days=1),
        importance=0.5,
    )
    stale = EpisodicMemory(
        content="Older event",
        created_at=now - timedelta(days=60),
        importance=0.5,
    )

    assert compute_decay_score(recent, now=now) > compute_decay_score(stale, now=now)


def test_episodic_memory_decays_faster_than_semantic_memory():
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    semantic = SemanticMemory(
        content="Stable fact",
        created_at=now - timedelta(days=60),
        importance=0.5,
    )
    episodic = EpisodicMemory(
        content="Old event",
        created_at=now - timedelta(days=60),
        importance=0.5,
    )

    assert compute_decay_score(episodic, now=now) < compute_decay_score(semantic, now=now)


def test_procedural_wilson_score_can_outweigh_age():
    now = datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc)
    older_high_quality = ProceduralMemory(
        content="Reliable deploy runbook",
        steps=["Build", "Deploy"],
        created_at=now - timedelta(days=120),
        success_count=95,
        failure_count=5,
    )
    newer_low_quality = ProceduralMemory(
        content="Unreliable deploy runbook",
        steps=["Build", "Deploy"],
        created_at=now - timedelta(days=7),
        success_count=1,
        failure_count=9,
    )

    assert older_high_quality.wilson_score > newer_low_quality.wilson_score
    assert compute_decay_score(older_high_quality, now=now) > compute_decay_score(
        newer_low_quality,
        now=now,
    )


def test_high_importance_semantic_memory_gets_half_life_floor_multiplier():
    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    baseline = SemanticMemory(
        content="Regular fact",
        created_at=now - timedelta(days=730),
        importance=0.79,
    )
    protected = SemanticMemory(
        content="High-importance fact",
        created_at=now - timedelta(days=730),
        importance=0.8,
    )

    assert compute_decay_score(protected, now=now) > compute_decay_score(baseline, now=now)


def test_semantic_half_life_boundary_score_is_exactly_point_two_five():
    now = datetime(2026, 12, 31, 0, 0, tzinfo=timezone.utc)
    record = SemanticMemory(
        content="Boundary fact",
        created_at=now - timedelta(days=365),
        importance=0.0,
    )

    assert compute_decay_score(record, now=now) == 0.25


def test_last_accessed_none_falls_back_to_created_at_and_naive_datetimes_are_utc():
    now = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
    aware = EpisodicMemory(
        content="Aware timestamp",
        created_at=now - timedelta(days=10),
        importance=0.5,
    )
    naive = EpisodicMemory(
        content="Naive timestamp",
        created_at=datetime(2026, 1, 21, 12, 0),
        importance=0.5,
    )

    assert math.isclose(compute_decay_score(aware, now=now), compute_decay_score(naive, now=now))
