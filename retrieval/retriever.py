from datetime import datetime, timezone

from events.bus import EventBus
from stores.base import BaseStore
from retrieval.ranking import RankedResult, rank_results

CANDIDATE_MULTIPLIER = 3  # over-fetch from each store to give the reranker room


class UnifiedRetriever:
    """Queries across memory stores, applies weighted ranking, and tracks access.

    Currently wraps SemanticStore only. When episodic and procedural stores
    are added, they join the _stores dict and fan-out happens automatically.
    """

    def __init__(self, stores: dict[str, BaseStore], event_bus: EventBus | None = None):
        self._stores = stores
        self._event_bus = event_bus

    def _emit_event(self, event_type: str, data: dict) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(event_type, data)

    def query(
        self,
        text: str,
        top_k: int = 5,
        memory_types: list[str] | None = None,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
    ) -> list[RankedResult]:
        # ── fan-out: query matching stores ──────────────────────────────────
        targets = self._stores
        if memory_types:
            targets = {k: v for k, v in self._stores.items() if k in memory_types}

        # Over-fetch so recency/importance can rescue items outside the
        # initial similarity slice.
        fetch_k = top_k * CANDIDATE_MULTIPLIER
        queried_memory_types = list(targets.keys())

        all_results = []
        for store in targets.values():
            all_results.extend(store.retrieve(text, top_k=fetch_k))

        self._emit_event(
            "memory.retrieved",
            {
                "query": text,
                "memory_types": queried_memory_types,
                "candidate_count": len(all_results),
                "top_similarity": max((score for _, score in all_results), default=None),
            },
        )

        # ── rank across all stores ──────────────────────────────────────────
        ranked = rank_results(
            all_results,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

        final = ranked[:top_k]

        self._emit_event(
            "memory.ranked",
            {
                "query": text,
                "results": [
                    {
                        "record_id": r.record.id,
                        "content": r.record.content,
                        "final_score": r.final_score,
                        "raw_similarity": r.raw_similarity,
                        "recency_score": r.recency_score,
                        "importance_score": r.importance_score,
                    }
                    for r in final
                ],
                "weights": {
                    "relevance": relevance_weight,
                    "recency": recency_weight,
                    "importance": importance_weight,
                },
            },
        )

        # ── update access tracking on returned records ──────────────────────
        now = datetime.now(timezone.utc)
        for r in final:
            store = self._stores.get(r.record.memory_type)
            if store:
                store.update_access(r.record.id)
                # Keep in-memory record in sync so callers see current values
                r.record.access_count += 1
                r.record.last_accessed_at = now
                self._emit_event(
                    "memory.accessed",
                    {
                        "record_id": r.record.id,
                        "memory_type": r.record.memory_type,
                        "access_count": r.record.access_count,
                    },
                )

        return final
