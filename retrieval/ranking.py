from datetime import datetime, timezone
from dataclasses import dataclass

from models.base import MemoryRecord


@dataclass
class RankedResult:
    record: MemoryRecord
    raw_similarity: float
    recency_score: float
    importance_score: float
    final_score: float


MIN_SPAN_SECONDS = 3600  # spans shorter than 1 hour are treated as "same time"


def rank_results(
    results: list[tuple[MemoryRecord, float]],
    relevance_weight: float = 0.4,
    recency_weight: float = 0.3,
    importance_weight: float = 0.3,
    now: datetime | None = None,
) -> list[RankedResult]:
    """Re-rank (record, similarity) pairs using a weighted combination of
    relevance, recency, and importance. Returns highest-score first."""

    if not results:
        return []

    now = now or datetime.now(timezone.utc)

    # ── normalise recency to 0-1 (min-max over candidate set) ───────────
    # Use last_accessed_at if set, otherwise created_at.
    # Newest candidate gets 1.0, oldest gets 0.0.
    # Single candidate or identical timestamps → all get 1.0.

    def _timestamp(record: MemoryRecord) -> datetime:
        ts = record.last_accessed_at or record.created_at
        # Normalise naive timestamps to UTC so subtraction never raises
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    ages = [(now - _timestamp(r)).total_seconds() for r, _ in results]
    min_age = min(ages)
    max_age = max(ages)
    span = max_age - min_age

    if span < MIN_SPAN_SECONDS:
        # All results are effectively the same age — no recency penalty
        recency_scores = [1.0] * len(results)
    else:
        # min_age → 1.0 (newest), max_age → 0.0 (oldest)
        recency_scores = [1.0 - (age - min_age) / span for age in ages]

    # ── compute final scores ────────────────────────────────────────────

    ranked = []
    for (record, similarity), recency in zip(results, recency_scores):
        final = (
            similarity * relevance_weight
            + recency * recency_weight
            + record.importance * importance_weight
        )
        ranked.append(RankedResult(
            record=record,
            raw_similarity=similarity,
            recency_score=recency,
            importance_score=record.importance,
            final_score=final,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
