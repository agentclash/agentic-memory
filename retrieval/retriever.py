from datetime import datetime, timezone
from typing import Any

from config import EMBEDDING_DIMENSIONS
from events.bus import EventBus
from models.base import MemoryRecord
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

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(event_type, data)

    def _emit_accessed(self, record: MemoryRecord) -> None:
        payload = {
            "record_id": record.id,
            "memory_type": record.memory_type,
            "access_count": record.access_count,
            "modality": record.modality,
        }
        if record.media_ref:
            payload["media_ref"] = record.media_ref
        self._emit_event("memory.accessed", payload)

    def _touch_records(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        now = datetime.now(timezone.utc)
        for record in records:
            store = self._stores.get(record.memory_type)
            if store is None:
                continue
            store.update_access(record.id)
            record.access_count += 1
            record.last_accessed_at = now
            self._emit_accessed(record)
        return records

    def _get_episodic_store(self):
        store = self._stores.get("episodic")
        if store is None:
            return None
        return store

    def _resolve_targets(self, memory_types: list[str] | None = None) -> dict[str, BaseStore]:
        if not memory_types:
            return self._stores
        return {key: value for key, value in self._stores.items() if key in memory_types}

    def _emit_ranked(
        self,
        *,
        final: list[RankedResult],
        relevance_weight: float,
        recency_weight: float,
        importance_weight: float,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = {
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
        }
        if extra_payload:
            payload.update(extra_payload)
        self._emit_event("memory.ranked", payload)

    def _rank_and_touch(
        self,
        *,
        all_results: list[tuple[MemoryRecord, float]],
        top_k: int,
        relevance_weight: float,
        recency_weight: float,
        importance_weight: float,
        ranked_payload: dict[str, Any] | None = None,
    ) -> list[RankedResult]:
        ranked = rank_results(
            all_results,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

        final = ranked[:top_k]
        self._emit_ranked(
            final=final,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            extra_payload=ranked_payload,
        )

        self._touch_records([r.record for r in final])
        return final

    def _validate_vector_dimensions(self, vector: list[float]) -> None:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected query vector dimension {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )

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
        targets = self._resolve_targets(memory_types)

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

        return self._rank_and_touch(
            all_results=all_results,
            top_k=top_k,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            ranked_payload={"query": text},
        )

    def query_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        memory_types: list[str] | None = None,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        metadata: dict[str, Any] | None = None,
    ) -> list[RankedResult]:
        self._validate_vector_dimensions(vector)

        targets = self._resolve_targets(memory_types)
        fetch_k = top_k * CANDIDATE_MULTIPLIER
        queried_memory_types = list(targets.keys())

        all_results = []
        for store in targets.values():
            all_results.extend(store.retrieve_by_vector(vector, top_k=fetch_k))

        retrieved_payload = {
            "query_type": "vector",
            "vector_dimensions": len(vector),
            "memory_types": queried_memory_types,
            "candidate_count": len(all_results),
            "top_similarity": max((score for _, score in all_results), default=None),
        }
        if metadata:
            retrieved_payload["query_metadata"] = metadata
        self._emit_event("memory.retrieved", retrieved_payload)

        ranked_payload = {
            "query_type": "vector",
            "vector_dimensions": len(vector),
        }
        if metadata:
            ranked_payload["query_metadata"] = metadata

        return self._rank_and_touch(
            all_results=all_results,
            top_k=top_k,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            ranked_payload=ranked_payload,
        )

    def query_recent(self, n: int) -> list[MemoryRecord]:
        store = self._get_episodic_store()
        records = []
        if store is not None and hasattr(store, "get_recent"):
            records = list(store.get_recent(n))

        self._emit_event(
            "memory.retrieved",
            {
                "query": "recent",
                "query_type": "recent",
                "memory_types": ["episodic"],
                "candidate_count": len(records),
                "top_similarity": None,
                "limit": n,
            },
        )
        return self._touch_records(records)

    def query_time_range(self, start: datetime, end: datetime) -> list[MemoryRecord]:
        store = self._get_episodic_store()
        records = []
        if store is not None and hasattr(store, "get_by_time_range"):
            records = list(store.get_by_time_range(start, end))

        self._emit_event(
            "memory.retrieved",
            {
                "query": "time_range",
                "query_type": "time_range",
                "memory_types": ["episodic"],
                "candidate_count": len(records),
                "top_similarity": None,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        return self._touch_records(records)
